/**
 * Retexture — Point-Filtered 3D Viewport with Dynamic Colored Lights & Halftone Shading
 */

(function () {
  let scene, camera, renderer, mesh, texture;
  let isInitialized = false;
  let currentGeometryType = "cube";
  let autoRotate = true;
  let coloredLightsEnabled = true;
  let halftone3DEnabled = false;
  let isDragging = false;
  let prevMousePos = { x: 0, y: 0 };
  let animTime = 0;

  let redLight, greenLight, blueLight, warmLight;
  let redOrb, greenOrb, blueOrb, warmOrb;
  let ambientLight, dirLight;

  window.init3DViewport = function () {
    if (isInitialized) return;
    const container = document.getElementById("canvas3dContainer");
    if (!container) return;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1e1f29);

    const aspect = container.clientWidth / container.clientHeight || 1;
    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    camera.position.set(0, 0, 3.5);

    renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    // Standard Lights
    ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);

    dirLight = new THREE.DirectionalLight(0xffffff, 0.45);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    // Dynamic Colored Point Lights (4 orbiting lights at 90° intervals)
    redLight = new THREE.PointLight(0xff2222, 2.8, 8.0);
    greenLight = new THREE.PointLight(0x22ff44, 2.8, 8.0);
    blueLight = new THREE.PointLight(0x2266ff, 2.8, 8.0);
    warmLight = new THREE.PointLight(0xffe0a0, 2.2, 8.0);

    const orbGeom = new THREE.SphereGeometry(0.09, 16, 16);
    redOrb = new THREE.Mesh(orbGeom, new THREE.MeshBasicMaterial({ color: 0xff3333 }));
    greenOrb = new THREE.Mesh(orbGeom, new THREE.MeshBasicMaterial({ color: 0x33ff55 }));
    blueOrb = new THREE.Mesh(orbGeom, new THREE.MeshBasicMaterial({ color: 0x4488ff }));
    warmOrb = new THREE.Mesh(orbGeom, new THREE.MeshBasicMaterial({ color: 0xffdd88 }));

    redLight.add(redOrb);
    greenLight.add(greenOrb);
    blueLight.add(blueOrb);
    warmLight.add(warmOrb);

    scene.add(redLight);
    scene.add(greenLight);
    scene.add(blueLight);
    scene.add(warmLight);

    // Restore saved 3D states from localStorage
    const savedMesh = localStorage.getItem("retexture_3d_mesh");
    if (savedMesh) currentGeometryType = savedMesh;
    const savedAuto = localStorage.getItem("retexture_3d_autorotate");
    if (savedAuto !== null) autoRotate = savedAuto === "true";
    const savedLights = localStorage.getItem("retexture_3d_colored_lights");
    if (savedLights !== null) coloredLightsEnabled = savedLights === "true";
    const savedHalftone = localStorage.getItem("retexture_3d_halftone");
    if (savedHalftone !== null) halftone3DEnabled = savedHalftone === "true";

    redLight.visible = coloredLightsEnabled;
    greenLight.visible = coloredLightsEnabled;
    blueLight.visible = coloredLightsEnabled;
    warmLight.visible = coloredLightsEnabled;
    ambientLight.intensity = coloredLightsEnabled ? 0.35 : 0.85;

    // Mesh
    createMesh(currentGeometryType);

    // Sync mesh selector buttons
    document.querySelectorAll(".mesh-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.mesh === currentGeometryType);
    });

    const chkAuto = document.getElementById("chkAutoRotate");
    if (chkAuto) chkAuto.checked = autoRotate;
    const chkLights = document.getElementById("chkColoredLights");
    if (chkLights) chkLights.checked = coloredLightsEnabled;
    const chkHalftone = document.getElementById("chkHalftone3D");
    if (chkHalftone) chkHalftone.checked = halftone3DEnabled;

    // Controls
    container.addEventListener("mousedown", (e) => {
      isDragging = true;
      prevMousePos = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mousemove", (e) => {
      if (!isDragging || !mesh) return;
      const deltaX = e.clientX - prevMousePos.x;
      const deltaY = e.clientY - prevMousePos.y;
      mesh.rotation.y += deltaX * 0.01;
      mesh.rotation.x += deltaY * 0.01;
      prevMousePos = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
    });

    container.addEventListener("wheel", (e) => {
      e.preventDefault();
      camera.position.z = Math.min(Math.max(camera.position.z + e.deltaY * 0.005, 1.2), 8.0);
    }, { passive: false });

    // Mesh selector buttons
    document.querySelectorAll(".mesh-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".mesh-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        currentGeometryType = btn.dataset.mesh;
        localStorage.setItem("retexture_3d_mesh", currentGeometryType);
        createMesh(currentGeometryType);
      });
    });

    if (chkAuto) {
      chkAuto.addEventListener("change", (e) => {
        autoRotate = e.target.checked;
        localStorage.setItem("retexture_3d_autorotate", autoRotate);
      });
    }

    if (chkLights) {
      chkLights.addEventListener("change", (e) => {
        coloredLightsEnabled = e.target.checked;
        redLight.visible = coloredLightsEnabled;
        greenLight.visible = coloredLightsEnabled;
        blueLight.visible = coloredLightsEnabled;
        warmLight.visible = coloredLightsEnabled;
        ambientLight.intensity = coloredLightsEnabled ? 0.35 : 0.85;
        localStorage.setItem("retexture_3d_colored_lights", coloredLightsEnabled);
      });
    }

    if (chkHalftone) {
      chkHalftone.addEventListener("change", (e) => {
        halftone3DEnabled = e.target.checked;
        localStorage.setItem("retexture_3d_halftone", halftone3DEnabled);
        createMesh(currentGeometryType);
      });
    }

    window.addEventListener("resize", () => {
      if (!container || !renderer || !camera) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });

    isInitialized = true;
    animate();
  };

  function createMesh(type) {
    if (mesh) {
      scene.remove(mesh);
      mesh.geometry.dispose();
    }

    let geom;
    switch (type) {
      case "sphere":
        geom = new THREE.SphereGeometry(1.2, 48, 36);
        break;
      case "cylinder":
        geom = new THREE.CylinderGeometry(1.0, 1.0, 2.0, 36);
        break;
      case "plane":
        geom = new THREE.PlaneGeometry(2.2, 2.2);
        break;
      case "cube":
      default:
        geom = new THREE.BoxGeometry(1.6, 1.6, 1.6);
        break;
    }

    let mat;
    if (halftone3DEnabled) {
      // Stylized Cel-Shaded Halftone material
      mat = new THREE.MeshToonMaterial({
        map: texture || null,
        side: THREE.DoubleSide,
      });
    } else {
      mat = new THREE.MeshStandardMaterial({
        map: texture || null,
        roughness: 0.65,
        metalness: 0.15,
        side: THREE.DoubleSide,
      });
    }

    mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);
  }

  window.update3DTexture = function (imageElement) {
    if (!imageElement) return;

    if (texture) texture.dispose();
    texture = new THREE.Texture(imageElement);
    const isNearest = window.state ? window.state.crispNearest !== false : true;
    texture.magFilter = isNearest ? THREE.NearestFilter : THREE.LinearFilter;
    texture.minFilter = isNearest ? THREE.NearestFilter : THREE.LinearMipmapLinearFilter;
    texture.generateMipmaps = !isNearest;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.needsUpdate = true;

    if (mesh && mesh.material) {
      mesh.material.map = texture;
      mesh.material.needsUpdate = true;
    }
  };

  window.set3DTextureFiltering = function (isNearest) {
    if (!texture) return;
    texture.magFilter = isNearest ? THREE.NearestFilter : THREE.LinearFilter;
    texture.minFilter = isNearest ? THREE.NearestFilter : THREE.LinearMipmapLinearFilter;
    texture.generateMipmaps = !isNearest;
    texture.needsUpdate = true;
    if (mesh && mesh.material) {
      mesh.material.needsUpdate = true;
    }
  };

  function animate() {
    requestAnimationFrame(animate);
    animTime += 0.02;

    if (autoRotate && mesh && !isDragging) {
      mesh.rotation.y += 0.006;
    }

    // Orbit 4 colored lights at 90° phase offsets
    if (coloredLightsEnabled && redLight) {
      const r = 2.4;
      const p0 = animTime;
      const p1 = animTime + Math.PI * 0.5;
      const p2 = animTime + Math.PI;
      const p3 = animTime + Math.PI * 1.5;
      redLight.position.set(Math.cos(p0) * r, Math.sin(p0 * 0.7) * 0.8, Math.sin(p0) * r);
      greenLight.position.set(Math.cos(p1) * r, Math.sin(p1 * 0.7) * 0.8, Math.sin(p1) * r);
      blueLight.position.set(Math.cos(p2) * r, Math.sin(p2 * 0.6) * 1.0, Math.sin(p2) * r);
      warmLight.position.set(Math.cos(p3) * r, Math.sin(p3 * 0.5) * 0.6, Math.sin(p3) * r);
    }

    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
  }
})();
