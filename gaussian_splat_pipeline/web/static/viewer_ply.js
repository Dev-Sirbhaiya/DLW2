// viewer_ply.js
// Three.js PLY viewer for Gaussian Splatting output
// Place this file in your static/ directory and include it in your HTML

import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

export function renderPLY(plyUrl, containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error('Container not found:', containerId);
        return;
    }

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 2;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Lighting
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(1, 1, 1).normalize();
    scene.add(light);

    // Load PLY
    const loader = new PLYLoader();
    loader.load(plyUrl, function (geometry) {
        geometry.computeVertexNormals();
        const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        animate();
    });

    // Controls (optional: add OrbitControls if desired)
    // import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
    // const controls = new OrbitControls(camera, renderer.domElement);

    function animate() {
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
    }
}
