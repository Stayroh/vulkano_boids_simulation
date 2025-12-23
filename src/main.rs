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

mod camera_controller;
use camera_controller::{CameraController, CameraState};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use rand::Rng;
use vulkano::command_buffer::{AutoCommandBufferBuilder, ClearColorImageInfo, ClearDepthStencilImageInfo, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::ClearDepthStencilValue;
use vulkano::image::{ImageCreateInfo, ImageType};
use vulkano::image::view::ImageView;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::pipeline::{PipelineBindPoint, PipelineCreateFlags, PipelineLayout};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, Subpass};
use vulkano::swapchain::{self, SwapchainPresentInfo};
use std::default;
use std::sync::Arc;
use vulkano::sync::{GpuFuture, now};
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
    sync::semaphore::{Semaphore, SemaphoreCreateInfo, SemaphoreType},
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
struct ComputePushConstants {
    delta_time: f32,
    radius_squared: f32,
    separation_scale: f32,
    alignment_scale: f32,
    cohesion_scale: f32,
    max_speed: f32,
    center_pull: f32,
    num_elements: u32,
}




const NUM_BOIDS: usize = 50_000;

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
    compute_pipeline: Arc<ComputePipeline>,
    boids_ssbo: Subbuffer<[Boid]>,
    descriptor_set_allocator:
        Arc<vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator>,
    command_buffer_allocator:
        Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
    camera_controller: CameraController,
    camera_buffer: Subbuffer<CameraState>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

impl GraphicsContext {
    fn update(&mut self, delta_time: f32) -> Result<()> {

        let compute_layout = self.compute_pipeline.layout().set_layouts().get(0).context("No descriptor set layout found")?;

        let compute_descriptor_set = vulkano::descriptor_set::DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            compute_layout.clone(),
            [vulkano::descriptor_set::WriteDescriptorSet::buffer(0, self.boids_ssbo.clone())],
            [],
        ).context("Failed to create descriptor set")?;

        let mut compute_builder = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.compute_queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        ).context("Failed to create command buffer builder")?;

        let push_constants = ComputePushConstants {
            delta_time,
            radius_squared: 1.5,
            separation_scale: 0.05,
            alignment_scale: 5.0,
            cohesion_scale: 20.0,
            max_speed: 500.0,
            center_pull: 0.5,
            num_elements: NUM_BOIDS as u32,
        };

        compute_builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .context("Failed to bind compute pipeline to command buffer")?
            .bind_descriptor_sets(
                vulkano::pipeline::PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                compute_descriptor_set,
            )
            .context("Failed to bind descriptor set to command buffer")?
            .push_constants(self.compute_pipeline.layout().clone(), 0, push_constants)
            .context("Failed to push constants to command buffer")?;

        unsafe {
            compute_builder
                .dispatch([NUM_BOIDS as u32 / 64 + 1, 1, 1])
                .context("Failed to dispatch compute shader")?;
        }

        let compute_command_buffer = compute_builder.build().context("Failed to build command buffer")?;

        
        let compute_future = vulkano::sync::now(self.device.clone())
            .then_execute(self.compute_queue.clone(), compute_command_buffer)
            .context("Failed to execute command buffer")?
            .then_signal_fence_and_flush()
            .context("Failed to signal fence and flush")?;

        compute_future.wait(None).context("Failed to wait for compute shader execution")?;

        

        let (image_index, _suboptimal, acquire_future) = swapchain::acquire_next_image(self.swapchain.clone(), None)
            .context("Failed to acquire next swapchain image")?;

        let dimensions = self.framebuffers[image_index as usize].attachments()[0].image().extent();

        println!("Rendering to image {} with dimensions {:?}", image_index, dimensions);
        let actual_window_size = self.window.inner_size();
        println!("Window size: {:?}", actual_window_size);

        
        let graphics_layout = self.graphics_pipeline.layout().set_layouts().get(0).context("No descriptor set layout found")?;

        let graphics_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            graphics_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.boids_ssbo.clone()),
                WriteDescriptorSet::buffer(1, self.camera_buffer.clone()),
            ],
            [],
        ).context("Failed to create graphics descriptor set")?;

        let mut graphics_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.queue_family_index(),
            command_buffer::CommandBufferUsage::OneTimeSubmit,
        ).context("Failed to create graphics auto command buffer builder")?;

        
        graphics_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None, None], //vec![
                        //Some([0.0, 0.0, 0.01, 1.0].into()),
                        //Some(vulkano::format::ClearValue::DepthStencil((1.0, 0)))
                    //],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone()
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .context("Failed to begin render pass")?
            .bind_pipeline_graphics(self.graphics_pipeline.clone())
            .context("Failed to bind graphics pipeline")?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.graphics_pipeline.layout().clone(),
                0,
                graphics_descriptor_set,
            )
            .context("Failed to bind graphics descriptor set")?;

        unsafe {
            graphics_builder
                .draw(
                    3,
                    NUM_BOIDS as u32,
                    0,
                    0,
                )
                .context("Failed to issue draw command")?;
        }

        graphics_builder
            .end_render_pass(Default::default())
            .context("Failed to end render pass")?;

        let graphics_command_buffer = graphics_builder.build().context("Failed to build graphics command buffer")?;

        let graphics_future = now(self.device.clone())
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), graphics_command_buffer)
            .context("Failed to execute graphics command buffer")?
            .then_swapchain_present(
                self.graphics_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush()
            .context("Failed to signal fence and flush for graphics")?;

        graphics_future.wait(None).context("Failed to wait for graphics execution")?;

        Ok(())
    }

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

        /*
        rfd::MessageDialog::new()
            .set_title("Vulkan Device Found")
            .set_description(&format!(
                "Using device: {}",
                physical_device.properties().device_name
            ))
            .set_level(rfd::MessageLevel::Info)
            .show();
        */

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
            .position(|(_i, q)| q.queue_flags.contains(QueueFlags::COMPUTE))
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

        println!("Using surface format: {:?}", image_format);

        let (swapchain, swapchain_images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_caps.min_image_count.max(2),
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                composite_alpha: surface_caps
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .context("No composite alpha mode supported")?,
                ..Default::default()
            },
        )
        .context("Failed to create swapchain")?;

        println!("Created swapchain with {} images", swapchain_images.len());

        let mut camera_controller = CameraController::new(
            70.0_f32.to_radians(),
            1.0,
            0.01,
            100.0
        );
        camera_controller.set_position([-35.0, 10.0, 0.0]);
        camera_controller.look_to([-1.0, 0.25, 0.0]);

        let boid_iter = BoidIter::new(NUM_BOIDS);

        let boids_ssbo = Buffer::from_iter(
            memory_allocator.clone(),
            vulkano::buffer::BufferCreateInfo {
                usage: vulkano::buffer::BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            boid_iter,
        )
        .context("Failed to create Boids Buffer")?;

        let camera_buffer = Buffer::from_data(
            memory_allocator.clone(),
            vulkano::buffer::BufferCreateInfo {
                usage: vulkano::buffer::BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            camera_controller.get_state(),
        ).context("Failed to create camera buffer")?;

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

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                },
                depth: {
                    format: vulkano::format::Format::D32_SFLOAT_S8_UINT,
                    samples: 1,
                    load_op: Load,
                    store_op: Store,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            },
        ).context("Failed to create render pass")?;

        let framebuffers = swapchain_images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone())
                    .context("Failed to create image view for framebuffer")?;
                let depth_view = ImageView::new_default(
                    Image::new(
                        memory_allocator.clone(),
                        ImageCreateInfo {
                            image_type: ImageType::Dim2d,
                            format: vulkano::format::Format::D32_SFLOAT_S8_UINT,
                            extent: image.extent(),
                            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSFER_DST,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                            ..Default::default()
                        },
                    ).context("Failed to create depth stencil image")?
                ).context("Failed to create depth stencil image view for framebuffers")?;

                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view, depth_view],
                        ..Default::default()
                    }
                ).context("Failed to create framebuffer")
            })
            .collect::<Result<Vec<_>>>()
            .context("Failed to create framebuffers")?;

        let command_buffer_allocator = Arc::new(
            vulkano::command_buffer::allocator::StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            ),
        );

        //Tell the GPU to fucking set those depth buffers to 1.0

        {
            let mut init_builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator.clone(),
                graphics_queue.queue_family_index(),
                command_buffer::CommandBufferUsage::OneTimeSubmit,
            ).context("Failed to create initialization command buffer builder")?;

            for fb in &framebuffers {
                init_builder
                   .clear_depth_stencil_image(ClearDepthStencilImageInfo {
                    clear_value: ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                    ..ClearDepthStencilImageInfo::image(fb.attachments().get(1).context("This should not happen")?.image().clone())
                   })
                   .context("Failed to record depth buffer clear command")?
                   .clear_color_image(ClearColorImageInfo {
                    clear_value: [0.5, 0.0, 0.01, 1.0].into(),
                    ..ClearColorImageInfo::image(fb.attachments().get(0).context("This should not happen")?.image().clone())
                   })
                   .context("Failed to record color buffer clear command")?;
            }

            let init_command_buffer = init_builder.build().context("Failed to build initialization command buffer")?;

            let init_future = now(device.clone())
                .then_execute(graphics_queue.clone(), init_command_buffer)
                .context("Failed to execute initialization command buffer")?
                .then_signal_fence_and_flush()
                .context("Failed to signal fence and flush for initialization")?;

            init_future.wait(None).context("Failed to wait for depth buffer initialization")?;
        }


        let graphics_pipeline = {
            
            let vs_module = vs::load(device.clone()).context("Failed to load vertex shader")?;
            let fs_module = fs::load(device.clone()).context("Failed to load fragment shader")?;

            let vertex_stage = PipelineShaderStageCreateInfo::new(
                vs_module
                    .entry_point("main")
                    .context("Error in vertex shader entry point")?,
            );
            
            let fragment_stage = PipelineShaderStageCreateInfo::new(
                fs_module
                    .entry_point("main")
                    .context("Error in fragment shader entry point")?,
            );

            let stages = [vertex_stage, fragment_stage];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .context("Failed to create pipeline layout info")?,
            )
            .context("Failed to create pipeline layout")?;

            let subpass = Subpass::from(render_pass.clone(), 0).context("Failed to crete subpass")?;
            
            let dimensions = framebuffers[0].attachments()[0].image().extent();

            let viewport = Viewport {
                    offset: [0.0, 0.0],
                    extent: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0..=1.0,
                };

            let graphics_pipeline_create_info = GraphicsPipelineCreateInfo {
                flags: PipelineCreateFlags::empty(),
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(VertexInputState::default()),
                input_assembly_state: Some(InputAssemblyState::default()),
                tessellation_state: None,
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState {
                    cull_mode: CullMode::None,
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)

            };

            GraphicsPipeline::new(
                device.clone(),
                None,
                graphics_pipeline_create_info,
            ).context("Failed to create graphics pipeline")?
        };

        let descriptor_set_allocator = Arc::new(
            vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator::new(
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
            compute_pipeline,
            boids_ssbo,
            descriptor_set_allocator,
            command_buffer_allocator,
            camera_controller,
            camera_buffer,
            graphics_pipeline,
            framebuffers
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
                if let Some(graphics_context) = &mut self.graphics_context {
                    let delta_time = 0.005; // Placeholder for ~60 FPS

                    if let Err(e) = graphics_context.update(delta_time) {
                        self.error = Some(e);
                        event_loop.exit();
                        return;
                    }
                } else {
                    self.error = Some(anyhow::anyhow!("Graphics context not initialized"));
                    event_loop.exit();
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
        let error_message = format!("{e:#}");
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
