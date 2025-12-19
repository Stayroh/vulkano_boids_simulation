
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

mod compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/compute.comp"
    }
}

use std::sync::Arc;

use anyhow::{Context, Result};
use vulkano::{
    device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    image::{Image, ImageUsage},
    instance::Instance,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::Window,
};

/// Holds all Vulkan-related state needed for rendering.
#[derive(Default)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

struct GraphicsContext {
    #[allow(dead_code)]
    instance: Arc<Instance>,
    #[allow(dead_code)]
    window: Arc<Window>,
    #[allow(dead_code)]
    surface: Arc<Surface>,
    #[allow(dead_code)]
    physical_device: Arc<PhysicalDevice>,
    #[allow(dead_code)]
    device: Arc<Device>,
    #[allow(dead_code)]
    queues: Vec<Arc<Queue>>,
    #[allow(dead_code)]
    swapchain: Arc<Swapchain>,
    #[allow(dead_code)]
    swapchain_images: Vec<Arc<Image>>,
    #[allow(dead_code)]
    pipeline: Arc<vulkano::pipeline::GraphicsPipeline>,
    #[allow(dead_code)]
    vertex_buffer: Arc<vulkano::buffer::BufferAccess + Send + Sync>,
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
            .set_description(&format!("Using device: {}", physical_device.properties().device_name))
            .set_level(rfd::MessageLevel::Info)
            .show();

        // Find a queue family that supports both graphics and presentation to our surface
        let graphics_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(i, q)| {
                q.queue_flags.contains(QueueFlags::GRAPHICS)
                    && physical_device.surface_support(i as u32, &surface).unwrap_or(false)
            })
            .context("No suitable graphics queue family found")? as u32;

        let compute_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(i, q)| q.queue_flags.contains(QueueFlags::COMPUTE))
            .context("No suitable compute queue family found")? as u32;

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

        let (device, queues_iter) = Device::new(physical_device.clone(), device_create_info)
            .context("Failed to create logical device")?;

        let graphics_queue = queues_iter.next().unwrap();
        let compute_queue = if compute_family_index == graphics_family_index {
            graphics_queue.clone()
        } else {
            queues_iter.next().unwrap()
        };

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

        // Create vertex buffer for a triangle
        let vertex_buffer = {
            use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::vertex_buffer(),
                false,
                [
                    Vertex { position: [-0.5, -0.5] },
                    Vertex { position: [0.0, 0.5] },
                    Vertex { position: [0.5, -0.25] },
                ]
                .into_iter(),
            )?
        };

        // Load shaders
        let vs = vs::load(device.clone())?;
        let fs = fs::load(device.clone())?;

        // Create graphics pipeline
        let pipeline = {
            use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
            use vulkano::render_pass::{RenderPass, Subpass, AttachmentDescription, LoadOp, StoreOp};
            use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
            use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
            use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};

            let render_pass = RenderPass::new(
                device.clone(),
                vulkano::render_pass::RenderPassCreateInfo {
                    attachments: vec![AttachmentDescription {
                        format: Some(image_format),
                        samples: 1,
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Store,
                        stencil_load_op: LoadOp::DontCare,
                        stencil_store_op: StoreOp::DontCare,
                        initial_layout: vulkano::image::ImageLayout::Undefined,
                        final_layout: vulkano::image::ImageLayout::PresentSrc,
                        ..Default::default()
                    }],
                    subpasses: vec![vulkano::render_pass::SubpassDescription {
                        color_attachments: vec![(0, vulkano::image::ImageLayout::ColorAttachmentOptimal)],
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            )?;

            GraphicsPipeline::start()
                .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
                .vertex_shader(vs.entry_point("main").unwrap(), ())
                .input_assembly_state(InputAssemblyState::new())
                .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                    Viewport {
                        origin: [0.0, 0.0],
                        dimensions: [window.inner_size().width as f32, window.inner_size().height as f32],
                        depth_range: 0.0..1.0,
                    },
                ]))
                .fragment_shader(fs.entry_point("main").unwrap(), ())
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())?
        };

        Ok(Self {
            instance,
            window,
            surface,
            physical_device,
            device,
            queues,
            swapchain,
            swapchain_images,
            pipeline,
            vertex_buffer,
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
                    use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
                    use vulkano::sync::{self, GpuFuture};
                    use vulkano::swapchain::{AcquireError, SwapchainAcquireFuture, SwapchainPresentInfo};
                    use vulkano::image::SwapchainImage;
                    use vulkano::sync::FlushError;

                    let queue = gc.queues[0].clone();
                    let swapchain = gc.swapchain.clone();
                    let images = &gc.swapchain_images;

                    let (image_index, suboptimal, acquire_future) = match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
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
                    ).unwrap();

                    builder
                        .begin_render_pass(
                            gc.pipeline.render_pass().clone(),
                            &images[image_index],
                            SubpassContents::Inline,
                            clear_values,
                        ).unwrap()
                        .bind_pipeline_graphics(gc.pipeline.clone())
                        .bind_vertex_buffers(0, gc.vertex_buffer.clone())
                        .draw(3, 1, 0, 0).unwrap()
                        .end_render_pass().unwrap();

                    let command_buffer = builder.build().unwrap();

                    let future = acquire_future
                        .then_execute(queue.clone(), command_buffer).unwrap()
                        .then_swapchain_present(
                            queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
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
