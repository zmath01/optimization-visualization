import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const extent = 2.9;
const heightScale = 0.40;
THREE.ColorManagement.enabled = true;

function createSurface(loss) {
    const grid = 48;
    const geometry = new THREE.PlaneGeometry(2 * extent, 2 * extent, grid, grid);
    geometry.rotateX(-Math.PI / 2);
    const positions = geometry.getAttribute("position");
    const colors = new Float32Array(positions.count * 3);
    const basin = new THREE.Color("#274981");
    const rim = new THREE.Color("#43b1ab");
    const color = new THREE.Color();
    let minimum = Infinity;
    let maximum = -Infinity;

    for (let i = 0; i < positions.count; i++) {
        const z = heightScale * loss(positions.getX(i), positions.getZ(i));
        positions.setY(i, z);
        minimum = Math.min(minimum, z);
        maximum = Math.max(maximum, z);
    }
    for (let i = 0; i < positions.count; i++) {
        color.copy(basin).lerp(rim, (positions.getY(i) - minimum) / (maximum - minimum));
        color.toArray(colors, 3 * i);
    }

    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    // Keep the plot grid on the surface, sampling every fourth row and column.
    const lines = new THREE.BufferGeometry();
    const points = [];
    const point = new THREE.Vector3();
    for (let row = 0; row <= grid; row += 4) {
        for (let column = 0; column < grid; column++) {
            for (const index of [
                row * (grid + 1) + column, row * (grid + 1) + column + 1,
                column * (grid + 1) + row, (column + 1) * (grid + 1) + row,
            ]) {
                point.fromBufferAttribute(positions, index);
                points.push(point.x, point.y + 0.01, point.z);
            }
        }
    }
    lines.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
    return new THREE.Group().add(
        new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
        })),
        new THREE.LineSegments(lines, new THREE.LineBasicMaterial({
            color: "#85b8d8",
            transparent: true,
            opacity: 0.35,
        })),
    );
}

export function createView(container, loss, totalSteps) {
    const palette = getComputedStyle(container);
    const color = (name) => palette.getPropertyValue(`--${name}`).trim();
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(color("plot"));
    scene.add(new THREE.HemisphereLight(0xf2f7ff, 0x2d3d61, 1.35), createSurface(loss));

    const camera = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 60);
    camera.position.set(6, 13, -8);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.setAttribute("aria-label", "3D loss surface with Adam’s trajectory");
    container.append(renderer.domElement);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enablePan = false;
    orbit.minZoom = 0.75;
    orbit.maxZoom = 2.2;
    orbit.maxPolarAngle = Math.PI / 2.7;
    orbit.target.set(0, 2.25, 0);
    renderer.domElement.style.touchAction = "pan-y";
    orbit.update();

    let drawFrame;
    function draw() {
        if (drawFrame !== undefined) return;
        drawFrame = requestAnimationFrame(() => {
            drawFrame = undefined;
            renderer.render(scene, camera);
        });
    }
    orbit.addEventListener("change", draw);
    new ResizeObserver(([entry]) => {
        const { width, height } = entry.contentRect;
        if (!width || !height) return;
        const aspect = width / height;
        const halfHeight = 4.4 / Math.min(1, aspect);
        camera.top = halfHeight;
        camera.bottom = -halfHeight;
        camera.left = -halfHeight * aspect;
        camera.right = halfHeight * aspect;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
        draw();
    }).observe(container);

    const trajectory = new THREE.BufferGeometry();
    const positions = new THREE.BufferAttribute(new Float32Array((totalSteps + 1) * 3), 3);
    trajectory.setAttribute("position", positions.setUsage(THREE.DynamicDrawUsage));
    const trailColor = color("path");
    const trail = new THREE.Line(trajectory, new THREE.LineBasicMaterial({ color: trailColor, depthTest: false }));
    const dots = new THREE.Points(trajectory, new THREE.PointsMaterial({
        color: trailColor, depthTest: false, size: 2.8, sizeAttenuation: false,
    }));
    trail.frustumCulled = false;
    dots.frustumCulled = false;
    const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.097, 12, 12),
        new THREE.MeshBasicMaterial({ color: trailColor, depthTest: false }),
    );
    scene.add(trail, dots, marker);

    const arrows = [
        { field: "grad", name: "gradient", scale: 0.34, lift: 0.65 },
        { field: "mHat", name: "moment", scale: 0.34, lift: 0.38 },
        { field: "delta", name: "update", scale: 1, lift: 0.13 },
    ].map((spec) => {
        const arrow = new THREE.ArrowHelper(undefined, undefined, 1, color(spec.name));
        arrow.line.material.depthTest = false;
        arrow.cone.material.depthTest = false;
        arrow.line.renderOrder = arrow.cone.renderOrder = 1;
        scene.add(arrow);
        return { ...spec, arrow };
    });
    const direction = new THREE.Vector3();
    let previousPoints;
    let drawnPoints = 0;

    return {
        update(points, step) {
            if (points !== previousPoints) {
                previousPoints = points;
                drawnPoints = 0;
            }
            for (; drawnPoints < points.length; drawnPoints++) {
                const [x, y] = points[drawnPoints];
                positions.setXYZ(drawnPoints, x, loss(x, y) * heightScale + 0.045, y);
            }
            positions.needsUpdate = true;
            trajectory.setDrawRange(0, points.length);
            marker.position.fromBufferAttribute(positions, points.length - 1);
            marker.position.y += 0.055;

            for (const { arrow, field, scale, lift } of arrows) {
                if (!step) {
                    arrow.visible = false;
                    continue;
                }
                const [x, y] = step.origin;
                const [dx, dy] = step[field];
                const length = direction.set(dx, 0, dy).length() * scale;
                arrow.visible = length > 0.015;
                if (arrow.visible) {
                    // Lift the diagnostic vectors separately so parallel arrows remain visible.
                    arrow.position.set(x, loss(x, y) * heightScale + lift, y);
                    arrow.setDirection(direction.normalize());
                    const tip = Math.min(0.16, length * 0.48);
                    arrow.setLength(length, tip, tip * 0.72);
                }
            }
            draw();
        },
    };
}
