// Shader modules generated at compile time
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/triangle.vert"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/triangle.frag"
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/compute.comp"
    }
}

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use rand::Rng;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, Subbuffer},
    command_buffer,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDevice,
    },
    image::{Image, ImageUsage},
    instance::Instance,
    memory::allocator::*,
    pipeline::{
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::Window,
};

/// Holds all Vulkan-related state needed for rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Boid {
    position: [f32; 3],
    velocity: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PushConstants {
    delta_time: f32,
    radius_squared: f32,
    separation_scale: f32,
    alignment_scale: f32,
    cohesion_scale: f32,
    max_speed: f32,
    num_elements: u32,
}

const NUM_BOIDS: usize = 1_000;

struct BoidIter {
    remaining: usize,
    rng: rand::rngs::ThreadRng,
}

impl BoidIter {
    fn new(n: usize) -> Self {
        Self {
            remaining: n,
            rng: rand::rng(),
        }
    }
}

impl Iterator for BoidIter {
    type Item = Boid;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;

        // Random position in unit cube
        let position = [rand::random(), rand::random(), rand::random()];

        // Random velocity inside unit sphere
        let velocity = loop {
            let v = [
                self.rng.random_range(-1.0..1.0),
                self.rng.random_range(-1.0..1.0),
                self.rng.random_range(-1.0..1.0),
            ];
            if v.iter().map(|x| x * x).sum::<f32>() <= 1.0 {
                break v;
            }
        };

        Some(Boid { position, velocity })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for BoidIter {
    fn len(&self) -> usize {
        self.remaining
    }
}

struct GraphicsContext {
    instance: Arc<Instance>,
    window: Arc<Window>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    compute_queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    compute_pipeline: Arc<ComputePipeline>,
    boids_ssbo: Subbuffer<[Boid]>,
    descriptor_set_allocator:
        Arc<vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator>,
    command_buffer_allocator:
        Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
}

impl GraphicsContext {
    fn new(
        window: Arc<Window>,
        required_extensions: vulkano::instance::InstanceExtensions,
    ) -> Result<Self> {
        let vulkan_library = vulkano::VulkanLibrary::new()
            .context("Failed to load Vulkan library - is Vulkan installed?")?;

        let instance = Instance::new(
            vulkan_library,
            vulkano::instance::InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .context("Failed to create Vulkan instance")?;

        let surface = Surface::from_window(instance.clone(), window.clone())
            .context("Failed to create window surface")?;

        let physical_device = instance
            .enumerate_physical_devices()
            .context("Failed to enumerate physical devices")?
            .next()
            .context("No physical device found")?;

        rfd::MessageDialog::new()
            .set_title("Vulkan Device Found")
            .set_description(&format!(
                "Using device: {}",
                physical_device.properties().device_name
            ))
            .set_level(rfd::MessageLevel::Info)
            .show();

        // Find a queue family that supports both graphics and presentation to our surface
        let graphics_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(i, q)| {
                q.queue_flags.contains(QueueFlags::GRAPHICS)
                    && physical_device
                        .surface_support(i as u32, &surface)
                        .unwrap_or(false)
            })
            .context("No suitable graphics queue family found")?
            as u32;

        let compute_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(i, q)| q.queue_flags.contains(QueueFlags::COMPUTE))
            .context("No suitable compute queue family found")?
            as u32;

        let mut queue_create_infos = vec![QueueCreateInfo {
            queue_family_index: graphics_family_index,
            ..Default::default()
        }];
        if compute_family_index != graphics_family_index {
            queue_create_infos.push(QueueCreateInfo {
                queue_family_index: compute_family_index,
                ..Default::default()
            });
        }

        let device_create_info = DeviceCreateInfo {
            queue_create_infos,
            enabled_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::empty()
            },
            ..Default::default()
        };

        let (device, mut queues_iter) = Device::new(physical_device.clone(), device_create_info)
            .context("Failed to create logical device")?;

        let graphics_queue = queues_iter.next().unwrap();
        let compute_queue = if compute_family_index == graphics_family_index {
            graphics_queue.clone()
        } else {
            queues_iter.next().unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        // Query surface capabilities to configure the swapchain properly
        let surface_caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .context("Failed to query surface capabilities")?;

        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .context("Failed to query surface formats")?
            .first()
            .context("No surface formats available")?
            .0;

        let (swapchain, swapchain_images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_caps.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_caps
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .context("No composite alpha mode supported")?,
                ..Default::default()
            },
        )
        .context("Failed to create swapchain")?;

        let boid_iter = BoidIter::new(NUM_BOIDS);

        let boids_ssbo = Buffer::from_iter(
            memory_allocator.clone(),
            vulkano::buffer::BufferCreateInfo {
                usage: vulkano::buffer::BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            boid_iter,
        )
        .context("Failed to create Boids Buffer")?;

        let compute_pipeline = {
            let cs_module = cs::load(device.clone()).context("Failed to load compute shader")?;

            let stage = PipelineShaderStageCreateInfo::new(
                cs_module
                    .entry_point("main")
                    .context("Error in compute shader entry point")?,
            );

            let layout = vulkano::pipeline::layout::PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .context("Failed to create pipeline layout info")?,
            )
            .context("Failed to create pipeline layout")?;

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .context("Failed to create compute pipeline")?
        };

        let descriptor_set_allocator = Arc::new(
            vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            ),
        );

        let command_buffer_allocator = Arc::new(
            vulkano::command_buffer::allocator::StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            ),
        );

        Ok(Self {
            instance,
            window,
            surface,
            physical_device,
            device,
            graphics_queue,
            compute_queue,
            swapchain,
            swapchain_images,
            compute_pipeline,
            boids_ssbo,
            descriptor_set_allocator,
            command_buffer_allocator,
        })
    }
}

/// Main application state that handles window events.
#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    graphics_context: Option<GraphicsContext>,
    /// Stores errors that occur during event handling.
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let result = (|| -> Result<()> {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .context("Failed to create window")?,
            );

            let required_extensions = Surface::required_extensions(event_loop)
                .context("Failed to get required surface extensions")?;

            self.graphics_context =
                Some(GraphicsContext::new(window.clone(), required_extensions)?);
            self.window = Some(window);

            Ok(())
        })();

        if let Err(e) = result {
            self.error = Some(e);
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                println!("Window close requested, exiting application.");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Draw the triangle using Vulkan
                if let Some(gc) = &self.graphics_context {
                    use vulkano::command_buffer::{
                        AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents,
                    };
                    use vulkano::image::SwapchainImage;
                    use vulkano::swapchain::{
                        AcquireError, SwapchainAcquireFuture, SwapchainPresentInfo,
                    };
                    use vulkano::sync::FlushError;
                    use vulkano::sync::{self, GpuFuture};

                    let queue = gc.queues[0].clone();
                    let swapchain = gc.swapchain.clone();
                    let images = &gc.swapchain_images;

                    let (image_index, suboptimal, acquire_future) =
                        match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                // Normally, recreate swapchain here
                                return;
                            }
                            Err(e) => {
                                eprintln!("Failed to acquire next image: {e}");
                                return;
                            }
                        };

                    let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into()];

                    let mut builder = AutoCommandBufferBuilder::primary(
                        gc.device.clone(),
                        queue.family(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();

                    builder
                        .begin_render_pass(
                            gc.pipeline.render_pass().clone(),
                            &images[image_index],
                            SubpassContents::Inline,
                            clear_values,
                        )
                        .unwrap()
                        .bind_pipeline_graphics(gc.pipeline.clone())
                        .bind_vertex_buffers(0, gc.vertex_buffer.clone())
                        .draw(3, 1, 0, 0)
                        .unwrap()
                        .end_render_pass()
                        .unwrap();

                    let command_buffer = builder.build().unwrap();

                    let future = acquire_future
                        .then_execute(queue.clone(), command_buffer)
                        .unwrap()
                        .then_swapchain_present(
                            queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(
                                swapchain.clone(),
                                image_index,
                            ),
                        )
                        .then_signal_fence_and_flush();

                    match future {
                        Ok(future) => {
                            // Wait for the GPU to finish
                            let _ = future.wait(None);
                        }
                        Err(FlushError::OutOfDate) => {
                            // Normally, recreate swapchain here
                        }
                        Err(e) => {
                            eprintln!("Failed to flush future: {e}");
                        }
                    }
                }
                // Request another redraw for continuous rendering
                if let Some(window) = &self.window {
                    window.request_redraw();
                } else {
                    self.error = Some(anyhow::anyhow!("Window not available for redraw"));
                    event_loop.exit();
                }
            }
            _ => (),
        }
    }
}

fn main() {
    if let Err(e) = run() {
        let error_message = format!("{e:#}"); // {:#} shows the full error chain
        eprintln!("Error: {error_message}");

        rfd::MessageDialog::new()
            .set_title("Fatal Error")
            .set_description(&error_message)
            .set_level(rfd::MessageLevel::Error)
            .show();

        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let event_loop = EventLoop::new().context("Failed to create event loop")?;
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    event_loop.run_app(&mut app).context("Event loop error")?;

    // Check if an error occurred during event handling
    if let Some(e) = app.error {
        return Err(e);
    }

    Ok(())
}
