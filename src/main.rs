use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent::*;
use winit::event_loop::*;
use winit::window::Window;

use vulkano::instance::Instance;
use vulkano::swapchain::Surface;


struct GraphicsContext {
    instance: Arc<Instance>,
    window: Arc<Window>,
    surface: Arc<Surface>,
}

impl GraphicsContext {
    fn new(window: Arc<Window>) -> Self {
        let vulkan_library = vulkano::VulkanLibrary::new().unwrap();
        let instance = vulkano::instance::Instance::new(vulkan_library, vulkano::instance::InstanceCreateInfo {
            enabled_extensions: vulkano::instance::InstanceExtensions {
                khr_surface: true,
                khr_wayland_surface: true,
                ..Default::default()
            },
            ..Default::default()
        }).unwrap();
        let surface = Surface::from_window(Arc::clone(&instance), Arc::clone(&window)).unwrap();
        println!("Vulkan instance and surface created successfully.");
        Self { instance, window, surface }
    }
}



#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    graphics_context: Option<GraphicsContext>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop
                .create_window(Window::default_attributes())
                .unwrap());
        self.graphics_context = Some(GraphicsContext::new(Arc::clone(&window)));
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _idfc: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            CloseRequested => {
                println!("Window close requested, exiting application.");
                event_loop.exit();
            }
            RedrawRequested => {
                println!("Window redraw requested.");
                // TODO: Actually render a frame here using Vulkan
                // For now, just acknowledge the redraw
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    let result = event_loop.run_app(&mut app);
    result.unwrap();
}
