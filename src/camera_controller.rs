use std::f32::consts::PI;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct CameraState {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub right: [f32; 3],
    pub up: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct CameraController {
    position: [f32; 3],
    forward: [f32; 3],
    up: [f32; 3],
    right: [f32; 3],
    
    // Projection parameters
    fov: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

impl CameraController {
    pub fn new(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let mut controller = Self {
            position: [0.0, 0.0, 0.0],
            forward: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            fov,
            aspect,
            near,
            far,
        };
        controller.update_vectors();
        controller
    }

    pub fn set_position(&mut self, position: [f32; 3]) {
        self.position = position;
    }

    pub fn set_aspect_ratio(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    pub fn look_to(&mut self, direction: [f32; 3]) {
        self.forward = Self::normalize(direction);
        self.update_vectors();
    }

    pub fn look_at(&mut self, target: [f32; 3]) {
        let direction = [
            target[0] - self.position[0],
            target[1] - self.position[1],
            target[2] - self.position[2],
        ];
        self.look_to(direction);
    }

    pub fn get_state(&self) -> CameraState {
        CameraState {
            view: self.compute_view_matrix(),
            proj: self.compute_projection_matrix(),
            right: self.right,
            up: self.up,
        }
    }

    fn update_vectors(&mut self) {
        self.forward = Self::normalize(self.forward);
        self.right = Self::normalize(Self::cross(self.forward, [0.0, 1.0, 0.0]));
        self.up = Self::normalize(Self::cross(self.right, self.forward));
    }

    fn compute_view_matrix(&self) -> [[f32; 4]; 4] {
        let f = self.forward;
        let r = self.right;
        let u = self.up;
        let p = self.position;

        [
            [r[0], u[0], -f[0], 0.0],
            [r[1], u[1], -f[1], 0.0],
            [r[2], u[2], -f[2], 0.0],
            [
                -Self::dot(r, p),
                -Self::dot(u, p),
                Self::dot(f, p),
                1.0,
            ],
        ]
    }

    fn compute_projection_matrix(&self) -> [[f32; 4]; 4] {
    let tan_half_fov = (self.fov / 2.0).tan();
    let range = self.far - self.near;
    
    [
        [1.0 / (self.aspect * tan_half_fov), 0.0, 0.0, 0.0],
        [0.0, -1.0 / tan_half_fov, 0.0, 0.0],  // Note the negative sign!
        [0.0, 0.0, self.far / range, 1.0],      // Changed for [0,1] depth
        [0.0, 0.0, -(self.far * self.near) / range, 0.0],
    ]
}

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 0.0 {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            v
        }
    }

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}

// Example usage
fn main() {
    let mut camera = CameraController::new(
        PI / 4.0,  // 45 degree FOV
        16.0 / 9.0, // aspect ratio
        0.1,        // near plane
        100.0,      // far plane
    );

    camera.set_position([0.0, 5.0, 10.0]);
    camera.look_at([0.0, 0.0, 0.0]);

    let state = camera.get_state();
    println!("Camera State:");
    println!("Right: {:?}", state.right);
    println!("Up: {:?}", state.up);
}