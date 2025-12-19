use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent::*;
use winit::event_loop::*;
use winit::window::Window;

use vulkano::instance::Instance;
use vulkano::swapchain::Surface;

// Use anyhow for easy error handling with context
use anyhow::{Context, Result};


struct GraphicsContext {
    instance: Arc<Instance>,
    window: Arc<Window>,
    surface: Arc<Surface>,
}

impl GraphicsContext {
    fn new(window: Arc<Window>, required_extensions: vulkano::instance::InstanceExtensions) -> Result<Self> {
        let vulkan_library = vulkano::VulkanLibrary::new()
            .context("Failed to load Vulkan library - is Vulkan installed?")?;
        let instance = vulkano::instance::Instance::new(vulkan_library, vulkano::instance::InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        }).context("Failed to create Vulkan instance")?;
        let surface = Surface::from_window(Arc::clone(&instance), Arc::clone(&window))
            .context("Failed to create window surface")?;
        println!("Vulkan instance and surface created successfully.");
        Ok(Self { instance, window, surface })
    }
}



#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    graphics_context: Option<GraphicsContext>,
    error: Option<anyhow::Error>,  // Store errors that occur during event handling
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let result = (|| -> Result<()> {
            let window = Arc::new(event_loop
                .create_window(Window::default_attributes())
                .context("Failed to create window")?);
            let required_extensions = Surface::required_extensions(event_loop)
                .context("Failed to get required surface extensions")?;
            self.graphics_context = Some(GraphicsContext::new(Arc::clone(&window), required_extensions)?);
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
    if let Err(e) = run() {
        // Show error dialog with full error chain
        let error_message = format!("{e:#}");  // {:#} shows the full error chain
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
