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
    /// Shaders can keep float32 / float64 denormals (otherwise they may flush to zero).
    pub denorm_preserve_f32: bool,
    pub denorm_preserve_f64: bool,
    pub subgroup_size: u32,
    pub max_shared_memory: u32,
    pub max_workgroup_count_x: u32,
    pub max_push_constants: u32,
    pub max_allocation_size: u64,
    pub max_buffer_size: u64,
    /// A memory type that is device-local *and* host-visible + coherent:
    /// on integrated GPUs this lets CPU and GPU share tensors without copies.
    pub unified_memory_type: Option<u32>,
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
    pub caps: Capabilities,
    memory: vk::PhysicalDeviceMemoryProperties,
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

            let caps = query_capabilities(&instance, physical_device, &props);
            for (ok, what) in [
                (caps.buffer_device_address, "bufferDeviceAddress"),
                (caps.shader_int64, "shaderInt64"),
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

            let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(caps.storage_buffer_16bit);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .scalar_block_layout(true)
                .timeline_semaphore(true)
                .vulkan_memory_model(caps.vulkan_memory_model)
                .vulkan_memory_model_device_scope(caps.vulkan_memory_model_device_scope)
                .shader_float16(caps.shader_float16)
                .shader_int8(caps.shader_int8)
                .storage_buffer8_bit_access(caps.storage_buffer_8bit);
            let core = vk::PhysicalDeviceFeatures::default()
                .shader_float64(caps.shader_float64)
                .shader_int64(true)
                .shader_int16(caps.shader_int16);
            let priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family).queue_priorities(&priorities);
            let device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_features(&core)
                .push_next(&mut f11)
                .push_next(&mut f12);
            let device = instance.create_device(physical_device, &device_info, None)?;
            let queue = device.get_device_queue(queue_family, 0);
            let memory = instance.get_physical_device_memory_properties(physical_device);

            Ok(Self { _entry: entry, instance, debug, physical_device, device, queue, queue_family, device_name, caps, memory })
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

unsafe fn query_capabilities(instance: &ash::Instance, pd: vk::PhysicalDevice, props: &vk::PhysicalDeviceProperties) -> Capabilities {
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12);
    instance.get_physical_device_features2(pd, &mut features);
    let core = features.features;

    let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
    let mut m3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut m4 = vk::PhysicalDeviceMaintenance4Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut p11).push_next(&mut p12).push_next(&mut m3).push_next(&mut m4);
    instance.get_physical_device_properties2(pd, &mut properties);

    let memory = instance.get_physical_device_memory_properties(pd);
    let unified = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let unified_memory_type = (0..memory.memory_type_count).find(|&i| memory.memory_types[i as usize].property_flags.contains(unified));

    let b = |v: vk::Bool32| v == vk::TRUE;
    Capabilities {
        api_version: props.api_version,
        shader_float64: b(core.shader_float64),
        shader_int64: b(core.shader_int64),
        shader_int16: b(core.shader_int16),
        shader_float16: b(f12.shader_float16),
        shader_int8: b(f12.shader_int8),
        buffer_device_address: b(f12.buffer_device_address),
        scalar_block_layout: b(f12.scalar_block_layout),
        vulkan_memory_model: b(f12.vulkan_memory_model),
        vulkan_memory_model_device_scope: b(f12.vulkan_memory_model_device_scope),
        timeline_semaphore: b(f12.timeline_semaphore),
        storage_buffer_16bit: b(f11.storage_buffer16_bit_access),
        storage_buffer_8bit: b(f12.storage_buffer8_bit_access),
        denorm_preserve_f32: b(p12.shader_denorm_preserve_float32),
        denorm_preserve_f64: b(p12.shader_denorm_preserve_float64),
        subgroup_size: p11.subgroup_size,
        max_shared_memory: props.limits.max_compute_shared_memory_size,
        max_workgroup_count_x: props.limits.max_compute_work_group_count[0],
        max_push_constants: props.limits.max_push_constants_size,
        max_allocation_size: m3.max_memory_allocation_size,
        max_buffer_size: m4.max_buffer_size,
        unified_memory_type,
    }
}

pub fn version_string(v: u32) -> String {
    format!("{}.{}.{}", vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v))
}
