//! Vulkan compute core: one device, one compute queue, buffers addressed by
//! raw device address, and push-constant-only compute pipelines.

use std::ffi::c_void;
use std::io::Cursor;

use ash::vk;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Device features this backend relies on, as reported by the driver.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    pub api_version: u32,
    pub shader_float64: bool,
    pub shader_int64: bool,
    pub shader_int16: bool,
    pub shader_float16: bool,
    pub buffer_device_address: bool,
    pub scalar_block_layout: bool,
    pub vulkan_memory_model: bool,
    pub storage_buffer_16bit: bool,
    pub storage_buffer_8bit: bool,
    pub subgroup_size: u32,
    pub max_shared_memory: u32,
    pub max_workgroup_count_x: u32,
    pub max_allocation_size: u64,
    pub max_buffer_size: u64,
    /// A memory type that is device-local *and* host-visible + coherent:
    /// on integrated GPUs this lets CPU and GPU share tensors without copies.
    pub unified_memory_type: Option<u32>,
}

pub struct Context {
    _entry: ash::Entry,
    instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub device_name: String,
    pub caps: Capabilities,
    memory: vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
}

impl Context {
    pub fn new() -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load()?;
            let app = vk::ApplicationInfo::default()
                .application_name(c"torch-vulkan")
                .api_version(vk::API_VERSION_1_3);
            let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)?;

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
                .vulkan_memory_model(caps.vulkan_memory_model)
                .shader_float16(caps.shader_float16)
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

            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let memory = instance.get_physical_device_memory_properties(physical_device);

            Ok(Self { _entry: entry, instance, physical_device, device, queue, queue_family, device_name, caps, memory, command_pool })
        }
    }

    pub fn memory_heaps(&self) -> &[vk::MemoryHeap] {
        &self.memory.memory_heaps[..self.memory.memory_heap_count as usize]
    }

    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        let types = &self.memory.memory_types[..self.memory.memory_type_count as usize];
        (0..types.len() as u32).find(|&i| bits & (1 << i) != 0 && types[i as usize].property_flags.contains(want))
    }

    /// A storage buffer that is mapped on the host and addressable from shaders.
    /// Uses unified (device-local + host-visible) memory when the GPU has it.
    pub fn create_buffer(&self, size: u64) -> Result<Buffer> {
        unsafe {
            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let type_index = self
                .memory_type(req.memory_type_bits, host | vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|| self.memory_type(req.memory_type_bits, host | vk::MemoryPropertyFlags::DEVICE_LOCAL))
                .or_else(|| self.memory_type(req.memory_type_bits, host))
                .ok_or("no host-visible memory type")?;
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(type_index).push_next(&mut flags),
                None,
            )?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;
            let mapped = self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?;
            let address = self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            Ok(Buffer { device: self.device.clone(), buffer, memory, mapped, size, address })
        }
    }

    /// A compute pipeline whose only inputs are `push_constant_size` bytes of push constants.
    pub fn create_kernel(&self, spirv: &[u8], push_constant_size: u32) -> Result<Kernel> {
        unsafe {
            let words = ash::util::read_spv(&mut Cursor::new(spirv))?;
            let module = self.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
            let range = vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(push_constant_size);
            let layout = self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(std::slice::from_ref(&range)),
                None,
            )?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let pipeline = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None)
                .map_err(|(_, e)| e)?[0];
            self.device.destroy_shader_module(module, None);
            Ok(Kernel { device: self.device.clone(), pipeline, layout })
        }
    }

    /// Records one dispatch, submits it, and blocks until the GPU is done and
    /// its writes are visible to the host.
    pub fn run<P: Copy>(&self, kernel: &Kernel, params: &P, groups: u32) -> Result<()> {
        unsafe {
            let d = &self.device;
            let cb = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(self.command_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
            )?[0];
            d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            let bytes = std::slice::from_raw_parts(params as *const P as *const u8, std::mem::size_of::<P>());
            d.cmd_push_constants(cb, kernel.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            d.cmd_dispatch(cb, groups, 1, 1);
            let to_host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[to_host], &[], &[]);
            d.end_command_buffer(cb)?;

            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&[cb])], fence)?;
            d.wait_for_fences(&[fence], true, u64::MAX)?;
            d.destroy_fence(fence, None);
            d.free_command_buffers(self.command_pool, &[cb]);
            Ok(())
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
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
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut p11);
    instance.get_physical_device_properties2(pd, &mut properties);
    let mut p13 = vk::PhysicalDeviceMaintenance4Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut p13);
    instance.get_physical_device_properties2(pd, &mut properties);
    let mut p3 = vk::PhysicalDeviceMaintenance3Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut p3);
    instance.get_physical_device_properties2(pd, &mut properties);

    let memory = instance.get_physical_device_memory_properties(pd);
    let unified = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let unified_memory_type =
        (0..memory.memory_type_count).find(|&i| memory.memory_types[i as usize].property_flags.contains(unified));

    let b = |v: vk::Bool32| v == vk::TRUE;
    Capabilities {
        api_version: props.api_version,
        shader_float64: b(core.shader_float64),
        shader_int64: b(core.shader_int64),
        shader_int16: b(core.shader_int16),
        shader_float16: b(f12.shader_float16),
        buffer_device_address: b(f12.buffer_device_address),
        scalar_block_layout: b(f12.scalar_block_layout),
        vulkan_memory_model: b(f12.vulkan_memory_model),
        storage_buffer_16bit: b(f11.storage_buffer16_bit_access),
        storage_buffer_8bit: b(f12.storage_buffer8_bit_access),
        subgroup_size: p11.subgroup_size,
        max_shared_memory: props.limits.max_compute_shared_memory_size,
        max_workgroup_count_x: props.limits.max_compute_work_group_count[0],
        max_allocation_size: p3.max_memory_allocation_size,
        max_buffer_size: p13.max_buffer_size,
        unified_memory_type,
    }
}

pub struct Buffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut c_void,
    pub size: u64,
    /// GPU virtual address of the first byte, for use as a shader pointer.
    pub address: vk::DeviceAddress,
}

impl Buffer {
    /// Host view of the buffer. The memory is coherent, but callers must not
    /// touch it while a GPU job that uses it is in flight.
    pub fn as_slice_mut<T: Copy>(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.mapped as *mut T, self.size as usize / std::mem::size_of::<T>()) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.unmap_memory(self.memory);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub struct Kernel {
    device: ash::Device,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub fn version_string(v: u32) -> String {
    format!("{}.{}.{}", vk::api_version_major(v), vk::api_version_minor(v), vk::api_version_patch(v))
}
