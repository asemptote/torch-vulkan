//! Instance, device and queue setup, capability queries, and optional
//! validation-layer reporting.

use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use ash::vk;

use crate::Result;

/// Device features this backend relies on, as reported by the driver.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    pub api_version: u32,
    pub shader_float64: bool,
    pub shader_int64: bool,
    pub shader_int16: bool,
    pub shader_float16: bool,
    pub shader_int8: bool,
    pub buffer_device_address: bool,
    pub scalar_block_layout: bool,
    pub vulkan_memory_model: bool,
    /// Device-scope barriers (factorization kernels order global memory
    /// between a workgroup's threads with them).
    pub vulkan_memory_model_device_scope: bool,
    pub timeline_semaphore: bool,
    pub storage_buffer_16bit: bool,
    pub storage_buffer_8bit: bool,
    /// VK_KHR_shader_float_controls (core in 1.2). SPIR-V 1.3 modules declare
    /// SPV_KHR_float_controls for their denormal modes; without it vkcore
    /// strips the modes, their capability and that extension.
    pub float_controls: bool,
    /// Shaders can keep float32 / float64 denormals (otherwise they may flush to zero).
    pub denorm_preserve_f32: bool,
    pub denorm_preserve_f64: bool,
    /// Shaders can ask for float64 NaN/inf/signed zero to be kept
    /// (SignedZeroInfNanPreserve): Mesa's ANV otherwise lowers 1.0 / NaN to 0.
    pub nan_preserve_f64: bool,
    pub subgroup_size: u32,
    /// Subgroup operations available to compute shaders (none if compute is
    /// not among the device's subgroup stages).
    pub subgroup_ops: vk::SubgroupFeatureFlags,
    /// VK_EXT_subgroup_size_control (core in 1.3) with full subgroups, for
    /// compute: a pipeline can then require a subgroup size.
    pub subgroup_size_control: bool,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    /// VK_KHR_cooperative_matrix with 16 x 16 x 16 float16 products into
    /// float32 at subgroup scope, in subgroups of 32 (gemmcm.slang), with
    /// what that kernel also needs: float16 arithmetic and storage, the
    /// Vulkan memory model with device scope, 38 KiB of shared memory.
    pub cooperative_matrix_f16: bool,
    /// The same for bfloat16 A and B (VK_KHR_shader_bfloat16's
    /// shaderBFloat16Type and shaderBFloat16CooperativeMatrix).
    pub cooperative_matrix_bf16: bool,
    pub max_shared_memory: u32,
    pub max_workgroup_count_x: u32,
    pub max_push_constants: u32,
    pub max_allocation_size: u64,
    pub max_buffer_size: u64,
    /// A memory type that is device-local *and* host-visible + coherent:
    /// on integrated GPUs this lets CPU and GPU share tensors without copies.
    pub unified_memory_type: Option<u32>,
}

impl Capabilities {
    /// The subgroup size for a kernel written for subgroups that prefers
    /// `preferred` lanes (gemmsg.slang: 16 for float32, 8 for float64, where a
    /// lane has more registers), or 0 where such kernels cannot run: they
    /// need size control with full subgroups, shuffles and broadcasts, and
    /// are instantiated for 8, 16 and 32 lanes. `TORCH_VK_SUBGROUP` overrides
    /// the preference (to test each size on one device).
    pub fn subgroup_size_for(&self, preferred: u32) -> u32 {
        let ops = vk::SubgroupFeatureFlags::SHUFFLE | vk::SubgroupFeatureFlags::BALLOT;
        if !self.subgroup_size_control || !self.subgroup_ops.contains(ops) {
            return 0;
        }
        let wanted = std::env::var("TORCH_VK_SUBGROUP").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(preferred);
        [wanted, preferred, 16, 8, 32]
            .into_iter()
            .find(|&s| [8, 16, 32].contains(&s) && s >= self.min_subgroup_size && s <= self.max_subgroup_size)
            .unwrap_or(0)
    }
}

/// One Vulkan device with one compute queue.
///
/// Set `TORCH_VK_VALIDATE=1` to enable the Khronos validation layer (with
/// synchronization and best-practices checks), or `TORCH_VK_VALIDATE=gpu`
/// for its GPU-assisted checks instead (every shader access through a buffer
/// device address is checked against the live allocations: much slower).
/// Every warning or error it reports is printed to stderr and counted in
/// [`validation_issues`].
pub struct Context {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub device_name: String,
    /// PCI vendor ID (0x13B5 Arm, 0x8086 Intel, 0x10DE NVIDIA, ...).
    pub vendor_id: u32,
    /// A discrete GPU (its own memory and power budget).
    pub discrete: bool,
    pub caps: Capabilities,
    memory: vk::PhysicalDeviceMemoryProperties,
    /// On Vulkan 1.1 devices, the extension entry points of the 1.2 core
    /// functions used (buffer device addresses, timeline semaphores).
    khr: Option<(ash::khr::buffer_device_address::Device, ash::khr::timeline_semaphore::Device)>,
}

static VALIDATION_ISSUES: AtomicUsize = AtomicUsize::new(0);
static LAST_VALIDATION_MESSAGE: Mutex<String> = Mutex::new(String::new());

/// Number of validation warnings/errors reported so far in this process.
pub fn validation_issues() -> usize {
    VALIDATION_ISSUES.load(Ordering::Relaxed)
}

/// The most recent validation warning/error ("" if none yet).
pub fn last_validation_message() -> String {
    LAST_VALIDATION_MESSAGE.lock().map(|m| m.clone()).unwrap_or_default()
}

/// Whether `TORCH_VK_VALIDATE` asks for the validation layer.
pub fn validation_requested() -> bool {
    std::env::var("TORCH_VK_VALIDATE").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Whether `TORCH_VK_VALIDATE=gpu` asks for GPU-assisted validation.
fn gpu_validation_requested() -> bool {
    std::env::var("TORCH_VK_VALIDATE").is_ok_and(|v| v == "gpu")
}

unsafe extern "system" fn on_validation_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let message = if data.is_null() || (*data).p_message.is_null() {
        "<no message>".into()
    } else {
        CStr::from_ptr((*data).p_message).to_string_lossy()
    };
    // The layer's notes on its own settings (which GPU-assisted checks it
    // turned off for this device) are not issues, nor is the loader's note
    // that it found the layer through a symlink.
    if message.contains("Information with the layer settings provided")
        || message.contains("validation is adjusting settings")
        || message.contains("was found to differ from OS loaded path")
    {
        return vk::FALSE;
    }
    VALIDATION_ISSUES.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut last) = LAST_VALIDATION_MESSAGE.lock() {
        *last = message.to_string();
    }
    eprintln!("[vulkan {severity:?}] {message}");
    vk::FALSE
}

fn messenger_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(on_validation_message))
}

impl Context {
    pub fn new() -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load()?;
            let app = vk::ApplicationInfo::default().application_name(c"torch-vulkan").api_version(vk::API_VERSION_1_3);

            let validate = validation_requested();
            let layers = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
            let extensions = [ash::ext::debug_utils::NAME.as_ptr()];
            let checks: &[vk::ValidationFeatureEnableEXT] = if gpu_validation_requested() {
                &[vk::ValidationFeatureEnableEXT::GPU_ASSISTED, vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT]
            } else {
                &[vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION, vk::ValidationFeatureEnableEXT::BEST_PRACTICES]
            };
            let mut features = vk::ValidationFeaturesEXT::default().enabled_validation_features(checks);
            let mut instance_messenger = messenger_info();
            let mut info = vk::InstanceCreateInfo::default().application_info(&app);
            if validate {
                info = info
                    .enabled_layer_names(&layers)
                    .enabled_extension_names(&extensions)
                    .push_next(&mut features)
                    .push_next(&mut instance_messenger);
            }
            let instance = entry.create_instance(&info, None)?;
            let debug = if validate {
                let utils = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let messenger = utils.create_debug_utils_messenger(&messenger_info(), None)?;
                Some((utils, messenger))
            } else {
                None
            };

            // Prefer a discrete GPU, then integrated, then anything.
            let devices = instance.enumerate_physical_devices()?;
            let rank = |pd: &vk::PhysicalDevice| match instance.get_physical_device_properties(*pd).device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 2,
            };
            let physical_device = *devices.iter().min_by_key(|pd| rank(pd)).ok_or("no Vulkan device found")?;
            let props = instance.get_physical_device_properties(physical_device);
            let device_name = props.device_name_as_c_str()?.to_string_lossy().into_owned();

            let exts = device_extensions(&instance, physical_device);
            let api = effective_api_version(props.api_version);
            let caps = query_capabilities(&entry, &instance, physical_device, &props, api, &exts);
            // (shaderInt64 and shaderFloat64 are optional: see kernels.rs)
            for (ok, what) in [
                (caps.buffer_device_address, "bufferDeviceAddress"),
                (caps.scalar_block_layout, "scalarBlockLayout"),
                (caps.timeline_semaphore, "timelineSemaphore"),
            ] {
                if !ok {
                    return Err(format!("{device_name} lacks required feature {what}").into());
                }
            }

            let queue_family = instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or("no compute queue")? as u32;

            // The features through their own structures (core since Vulkan
            // 1.2, extensions before), so that Vulkan 1.1 devices work too;
            // on those the extensions are enabled by name.
            let mut bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
            let mut sbl = vk::PhysicalDeviceScalarBlockLayoutFeatures::default().scalar_block_layout(true);
            let mut tls = vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
            let mut vmm = vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
                .vulkan_memory_model(caps.vulkan_memory_model)
                .vulkan_memory_model_device_scope(caps.vulkan_memory_model_device_scope);
            let mut f16i8 =
                vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(caps.shader_float16).shader_int8(caps.shader_int8);
            let mut s8 = vk::PhysicalDevice8BitStorageFeatures::default().storage_buffer8_bit_access(caps.storage_buffer_8bit);
            let mut s16 = vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(caps.storage_buffer_16bit);
            let core = vk::PhysicalDeviceFeatures::default()
                .shader_float64(caps.shader_float64)
                .shader_int64(caps.shader_int64)
                .shader_int16(caps.shader_int16);
            let has = |e: &CStr| exts.iter().any(|x| x.as_c_str() == e);
            let mut names: Vec<*const std::ffi::c_char> = Vec::new();
            if api < vk::API_VERSION_1_2 {
                for (e, wanted) in [
                    (ash::khr::buffer_device_address::NAME, true),
                    (ash::ext::scalar_block_layout::NAME, true),
                    (ash::khr::timeline_semaphore::NAME, true),
                    (ash::khr::vulkan_memory_model::NAME, caps.vulkan_memory_model),
                    (ash::khr::shader_float16_int8::NAME, caps.shader_float16 || caps.shader_int8),
                    (ash::khr::_8bit_storage::NAME, caps.storage_buffer_8bit),
                    (ash::khr::shader_float_controls::NAME, caps.float_controls),
                ] {
                    if wanted && has(e) {
                        names.push(e.as_ptr());
                    }
                }
            }
            if caps.subgroup_size_control && api < vk::API_VERSION_1_3 {
                names.push(ash::ext::subgroup_size_control::NAME.as_ptr());
            }
            if caps.cooperative_matrix_f16 || caps.cooperative_matrix_bf16 {
                names.push(ash::khr::cooperative_matrix::NAME.as_ptr());
            }
            if caps.cooperative_matrix_bf16 {
                names.push(SHADER_BFLOAT16_NAME.as_ptr());
            }
            let priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&priorities);
            let mut device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&names)
                .enabled_features(&core)
                .push_next(&mut bda)
                .push_next(&mut sbl)
                .push_next(&mut tls)
                .push_next(&mut s16);
            if caps.vulkan_memory_model {
                device_info = device_info.push_next(&mut vmm);
            }
            if caps.shader_float16 || caps.shader_int8 {
                device_info = device_info.push_next(&mut f16i8);
            }
            if caps.storage_buffer_8bit {
                device_info = device_info.push_next(&mut s8);
            }
            let mut ssc = vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
                .subgroup_size_control(true)
                .compute_full_subgroups(true);
            if caps.subgroup_size_control {
                device_info = device_info.push_next(&mut ssc);
            }
            let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            if caps.cooperative_matrix_f16 || caps.cooperative_matrix_bf16 {
                device_info = device_info.push_next(&mut coop);
            }
            let mut bf16 = ShaderBfloat16Features {
                shader_bfloat16_type: vk::TRUE,
                shader_bfloat16_cooperative_matrix: vk::TRUE,
                ..ShaderBfloat16Features::default()
            };
            if caps.cooperative_matrix_bf16 {
                device_info = device_info.push_next(&mut bf16);
            }
            let device = instance.create_device(physical_device, &device_info, None)?;
            let queue = device.get_device_queue(queue_family, 0);
            let memory = instance.get_physical_device_memory_properties(physical_device);

            let khr = (api < vk::API_VERSION_1_2).then(|| {
                (ash::khr::buffer_device_address::Device::new(&instance, &device),
                 ash::khr::timeline_semaphore::Device::new(&instance, &device))
            });
            Ok(Self { _entry: entry, instance, debug, physical_device, device, queue, queue_family, device_name, vendor_id: props.vendor_id, discrete: props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU, caps, memory, khr })
        }
    }

    /// vkGetBufferDeviceAddress (core 1.2, or its KHR extension on 1.1).
    ///
    /// # Safety
    ///
    /// The same as for the Vulkan call: the handles must belong to this device and be valid.
    pub unsafe fn buffer_device_address(&self, buffer: vk::Buffer) -> u64 {
        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        match &self.khr {
            Some((bda, _)) => bda.get_buffer_device_address(&info),
            None => self.device.get_buffer_device_address(&info),
        }
    }

    /// vkGetSemaphoreCounterValue (core 1.2, or KHR on 1.1).
    ///
    /// # Safety
    ///
    /// The same as for the Vulkan call: the handles must belong to this device and be valid.
    pub unsafe fn semaphore_counter_value(&self, semaphore: vk::Semaphore) -> std::result::Result<u64, vk::Result> {
        match &self.khr {
            Some((_, tls)) => tls.get_semaphore_counter_value(semaphore),
            None => self.device.get_semaphore_counter_value(semaphore),
        }
    }

    /// vkWaitSemaphores (core 1.2, or KHR on 1.1).
    ///
    /// # Safety
    ///
    /// The same as for the Vulkan call: the handles must belong to this device and be valid.
    pub unsafe fn wait_semaphores(&self, info: &vk::SemaphoreWaitInfo, timeout: u64) -> std::result::Result<(), vk::Result> {
        match &self.khr {
            Some((_, tls)) => tls.wait_semaphores(info, timeout),
            None => self.device.wait_semaphores(info, timeout),
        }
    }

    pub fn memory_heaps(&self) -> &[vk::MemoryHeap] {
        &self.memory.memory_heaps[..self.memory.memory_heap_count as usize]
    }

    /// First memory type allowed by `bits` that has all of `want`.
    pub(crate) fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        let types = &self.memory.memory_types[..self.memory.memory_type_count as usize];
        (0..types.len() as u32).find(|&i| bits & (1 << i) != 0 && types[i as usize].property_flags.contains(want))
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            if let Some((utils, messenger)) = self.debug.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn device_extensions(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Vec<std::ffi::CString> {
    instance
        .enumerate_device_extension_properties(pd)
        .unwrap_or_default()
        .iter()
        .filter_map(|e| e.extension_name_as_c_str().ok().map(|c| c.to_owned()))
        .collect()
}

/// The device's API version, or 1.1 with `TORCH_VK_API=1.1` (to exercise the
/// Vulkan 1.1 path, with extensions by name, on a newer device).
fn effective_api_version(device: u32) -> u32 {
    match std::env::var("TORCH_VK_API").as_deref() {
        Ok("1.1") => device.min(vk::API_VERSION_1_1),
        _ => device,
    }
}

/// VK_KHR_shader_bfloat16 (newer than ash 0.38's headers): the extension's
/// name, its feature structure and the bfloat16 component type.
const SHADER_BFLOAT16_NAME: &CStr = c"VK_KHR_shader_bfloat16";
const COMPONENT_TYPE_BFLOAT16: vk::ComponentTypeKHR = vk::ComponentTypeKHR::from_raw(1_000_141_000);

#[repr(C)]
#[derive(Clone, Copy)]
struct ShaderBfloat16Features {
    s_type: vk::StructureType,
    p_next: *mut std::ffi::c_void,
    shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    shader_bfloat16_cooperative_matrix: vk::Bool32,
}

impl Default for ShaderBfloat16Features {
    fn default() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(1_000_141_000),
            p_next: std::ptr::null_mut(),
            shader_bfloat16_type: vk::FALSE,
            shader_bfloat16_dot_product: vk::FALSE,
            shader_bfloat16_cooperative_matrix: vk::FALSE,
        }
    }
}

// (a structure that leads with sType and pNext, as the chains require)
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for ShaderBfloat16Features {}
unsafe impl vk::ExtendsDeviceCreateInfo for ShaderBfloat16Features {}

/// Whether `TORCH_VK_DISABLE` (comma-separated: int64, float64, int16,
/// float16, int8, storage16, storage8, denorm, subgroups, coopmat, bfloat16) names
/// `what`: the feature is then treated as missing, to test the paths for
/// devices without it (Mali GPUs have neither int64 nor float64; "subgroups"
/// hides subgroup size control, so no kernel written for subgroups runs;
/// "coopmat" hides cooperative matrices).
fn disabled(what: &str) -> bool {
    std::env::var("TORCH_VK_DISABLE").is_ok_and(|v| v.split(',').any(|x| x.trim() == what))
}

unsafe fn query_capabilities(
    entry: &ash::Entry,
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    props: &vk::PhysicalDeviceProperties,
    api: u32,
    exts: &[std::ffi::CString],
) -> Capabilities {
    let has = |e: &CStr| exts.iter().any(|x| x.as_c_str() == e);
    // (each structure only where the device knows it: core in 1.2, or its extension)
    let promoted = |e: &CStr| api >= vk::API_VERSION_1_2 || has(e);
    let mut bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    let mut sbl = vk::PhysicalDeviceScalarBlockLayoutFeatures::default();
    let mut tls = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
    let mut vmm = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
    let mut f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut s8 = vk::PhysicalDevice8BitStorageFeatures::default();
    let mut s16 = vk::PhysicalDevice16BitStorageFeatures::default();
    let size_control = api >= vk::API_VERSION_1_3 || has(ash::ext::subgroup_size_control::NAME);
    let mut ssc = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
    let has_coop = has(ash::khr::cooperative_matrix::NAME);
    let mut coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let has_bf16 = has(SHADER_BFLOAT16_NAME);
    let mut bf16 = ShaderBfloat16Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut s16);
    if has_coop {
        features = features.push_next(&mut coop);
    }
    if has_bf16 {
        features = features.push_next(&mut bf16);
    }
    if size_control {
        features = features.push_next(&mut ssc);
    }
    if promoted(ash::khr::buffer_device_address::NAME) {
        features = features.push_next(&mut bda);
    }
    if promoted(ash::ext::scalar_block_layout::NAME) {
        features = features.push_next(&mut sbl);
    }
    if promoted(ash::khr::timeline_semaphore::NAME) {
        features = features.push_next(&mut tls);
    }
    if promoted(ash::khr::vulkan_memory_model::NAME) {
        features = features.push_next(&mut vmm);
    }
    if promoted(ash::khr::shader_float16_int8::NAME) {
        features = features.push_next(&mut f16i8);
    }
    if promoted(ash::khr::_8bit_storage::NAME) {
        features = features.push_next(&mut s8);
    }
    instance.get_physical_device_features2(pd, &mut features);
    let core = features.features;

    let float_controls = promoted(ash::khr::shader_float_controls::NAME);
    let maintenance4 = api >= vk::API_VERSION_1_3 || has(ash::khr::maintenance4::NAME);
    let mut sub = vk::PhysicalDeviceSubgroupProperties::default();
    let mut fc = vk::PhysicalDeviceFloatControlsProperties::default();
    let mut m3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut m4 = vk::PhysicalDeviceMaintenance4Properties::default();
    let mut sscp = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut sub).push_next(&mut m3);
    if float_controls {
        properties = properties.push_next(&mut fc);
    }
    if maintenance4 {
        properties = properties.push_next(&mut m4);
    }
    if size_control {
        properties = properties.push_next(&mut sscp);
    }
    instance.get_physical_device_properties2(pd, &mut properties);
    let subgroup_ops = if sub.supported_stages.contains(vk::ShaderStageFlags::COMPUTE) {
        sub.supported_operations
    } else {
        vk::SubgroupFeatureFlags::empty()
    };
    let subgroup_size_control = size_control
        && ssc.subgroup_size_control == vk::TRUE
        && ssc.compute_full_subgroups == vk::TRUE
        && sscp.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE)
        && !disabled("subgroups");
    let (min_size, max_size) =
        if subgroup_size_control { (sscp.min_subgroup_size, sscp.max_subgroup_size) } else { (sub.subgroup_size, sub.subgroup_size) };
    // The cooperative-matrix shapes the device multiplies: gemmcm.slang's is
    // 16 x 16 x 16, float16 (or bfloat16) A and B into float32 C, one
    // subgroup each.
    let shapes = if has_coop && coop.cooperative_matrix == vk::TRUE {
        let ext = ash::khr::cooperative_matrix::Instance::new(entry, instance);
        ext.get_physical_device_cooperative_matrix_properties(pd)
            .unwrap_or_default()
            .iter()
            .filter(|c| {
                (c.m_size, c.n_size, c.k_size) == (16, 16, 16)
                    && c.a_type == c.b_type
                    && c.c_type == vk::ComponentTypeKHR::FLOAT32
                    && c.result_type == vk::ComponentTypeKHR::FLOAT32
                    && c.scope == vk::ScopeKHR::SUBGROUP
            })
            .map(|c| c.a_type)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let coop_f16_shape = shapes.contains(&vk::ComponentTypeKHR::FLOAT16);
    let coop_bf16_shape = shapes.contains(&COMPONENT_TYPE_BFLOAT16);


    let memory = instance.get_physical_device_memory_properties(pd);
    let unified = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let unified_memory_type = (0..memory.memory_type_count).find(|&i| memory.memory_types[i as usize].property_flags.contains(unified));

    let b = |v: vk::Bool32| v == vk::TRUE;
    // what gemmcm.slang needs besides its fragment types
    let coop_base = subgroup_size_control
        && (min_size..=max_size).contains(&32)
        && props.limits.max_compute_shared_memory_size >= 38912
        && b(s16.storage_buffer16_bit_access)
        && b(vmm.vulkan_memory_model)
        && b(vmm.vulkan_memory_model_device_scope)
        && !disabled("storage16")
        && !disabled("coopmat");
    Capabilities {
        api_version: props.api_version,
        shader_float64: b(core.shader_float64) && !disabled("float64"),
        shader_int64: b(core.shader_int64) && !disabled("int64"),
        shader_int16: b(core.shader_int16) && !disabled("int16"),
        shader_float16: b(f16i8.shader_float16) && !disabled("float16"),
        shader_int8: b(f16i8.shader_int8) && !disabled("int8"),
        buffer_device_address: b(bda.buffer_device_address),
        scalar_block_layout: b(sbl.scalar_block_layout),
        vulkan_memory_model: b(vmm.vulkan_memory_model),
        vulkan_memory_model_device_scope: b(vmm.vulkan_memory_model_device_scope),
        timeline_semaphore: b(tls.timeline_semaphore),
        storage_buffer_16bit: b(s16.storage_buffer16_bit_access) && !disabled("storage16"),
        storage_buffer_8bit: b(s8.storage_buffer8_bit_access) && !disabled("storage8"),
        float_controls,
        denorm_preserve_f32: float_controls && b(fc.shader_denorm_preserve_float32) && !disabled("denorm"),
        denorm_preserve_f64: float_controls && b(fc.shader_denorm_preserve_float64) && !disabled("denorm"),
        nan_preserve_f64: float_controls && b(fc.shader_signed_zero_inf_nan_preserve_float64) && !disabled("nanpreserve"),
        subgroup_size: sub.subgroup_size,
        subgroup_ops,
        subgroup_size_control,
        min_subgroup_size: min_size,
        max_subgroup_size: max_size,
        cooperative_matrix_f16: coop_base && coop_f16_shape && b(f16i8.shader_float16) && !disabled("float16"),
        cooperative_matrix_bf16: coop_base
            && coop_bf16_shape
            && has_bf16
            && b(bf16.shader_bfloat16_type)
            && b(bf16.shader_bfloat16_cooperative_matrix)
            && !disabled("bfloat16"),
        max_shared_memory: props.limits.max_compute_shared_memory_size,
        max_workgroup_count_x: props.limits.max_compute_work_group_count[0],
        max_push_constants: props.limits.max_push_constants_size,
        max_allocation_size: m3.max_memory_allocation_size,
        // (without maintenance4: no stated limit beyond one allocation's)
        max_buffer_size: if maintenance4 { m4.max_buffer_size } else { m3.max_memory_allocation_size },
        unified_memory_type,
    }
}

pub fn version_string(v: u32) -> String {
    format!("{}.{}.{}", vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v))
}
