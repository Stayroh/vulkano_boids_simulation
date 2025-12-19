fn main() {
    println!("cargo:rerun-if-changed=src/triangle.vert");
    println!("cargo:rerun-if-changed=src/triangle.frag");
    println!("cargo:rerun-if-changed=src/compute.comp");
}
