# Transition State Theory Visualizer

This project implements an interactive teaching tool for exploring transition state theory concepts on top of a configurable two-dimensional potential energy surface (PES). The backend is a lightweight Python HTTP server, while the frontend uses HTML and [three.js](https://threejs.org/) for real-time 3D rendering.

## Features

- **Gaussian Potential Builder** – combine up to three Gaussian features via sliders to create custom PES landscapes.
- **Circular States** – visualise two metastable states (A and B) as circular regions on the surface.
- **Nudged Elastic Band (NEB)** – compute a minimum-energy path between states and highlight the transition state along the path.
- **Transition Boundary Estimation** – approximate the dividing surface using an iso-energy contour around the transition state energy.
- **Umbrella Sampling** – perform simple umbrella sampling around each NEB image and chart the resulting free-energy profile.

## Getting started

1. Launch the development server (no external dependencies are required):

   ```bash
   python -m app.server
   ```

2. Open the application in your browser at [http://localhost:5000](http://localhost:5000).

Adjust the Gaussian sliders to reshape the PES. The NEB pathway, transition state and umbrella sampling analysis refresh automatically.

## Testing

Run the unit tests with:

```bash
pytest
```
