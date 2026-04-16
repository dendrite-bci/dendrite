---
layout: page
sidebar: false
---

<script setup>
import { withBase } from 'vitepress'
</script>

<div class="landing">
  <header class="hero">
    <h1 class="title">Dendrite</h1>
    <p class="subtitle">Open-source brain-computer interface application</p>
    <div class="actions">
      <a class="btn" :href="withBase('/quickstart')">Get Started</a>
    </div>
  </header>
  <section class="features">
    <div class="feature">
      <h3>Data Acquisition</h3>
      <p>Acquire EEG, EMG, and event markers over LSL. Record to HDF5, replay past sessions as live streams.</p>
    </div>
    <div class="feature">
      <h3>Signal Processing & ML</h3>
      <p>Three composable BCI modes with online preprocessing, decoder training, and real-time classification.</p>
    </div>
    <div class="feature">
      <h3>Task Integration</h3>
      <p>Output predictions to external applications over LSL, ROS2, TCP, or ZeroMQ.</p>
    </div>
  </section>
</div>

<style scoped>
.landing {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 24px;
}

.landing .hero {
  padding: 11rem 0 5rem;
  text-align: center;
}

.landing .title {
  font-size: 3.5rem;
  font-weight: 700;
  margin-bottom: 1rem;
  letter-spacing: -0.02em;
  line-height: 1.1;
}

.landing .subtitle {
  font-size: 1.25rem;
  color: var(--vp-c-text-2);
  margin: 0 auto 2.5rem;
  max-width: 700px;
  line-height: 1.5;
}

.landing .actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
}

.landing .btn {
  background: var(--vp-c-brand-1);
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 0.75rem 2rem;
  font-weight: 500;
  font-size: 1rem;
  text-decoration: none;
  transition: all 150ms ease;
  box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35);
}

.landing .btn:hover {
  background: var(--vp-c-brand-2);
  color: #fff;
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(59, 130, 246, 0.45);
}

:global(html.dark) .landing .btn {
  background: #3b82f6;
  box-shadow: 0 4px 14px rgba(59, 130, 246, 0.5);
}

:global(html.dark) .landing .btn:hover {
  background: #60a5fa;
  box-shadow: 0 6px 20px rgba(96, 165, 250, 0.55);
}

.landing .features {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0;
  padding: 3rem 0;
}

.landing .feature {
  padding: 1.5rem 2rem;
  border-left: 1px solid var(--vp-c-divider);
}

.landing .feature:first-child {
  border-left: none;
}

.landing .feature h3 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
}

.landing .feature p {
  font-size: 0.95rem;
  line-height: 1.6;
  color: var(--vp-c-text-2);
  margin: 0;
}

@media (max-width: 768px) {
  .landing .title { font-size: 2.5rem; }
  .landing .subtitle { font-size: 1.1rem; }
  .landing .features { grid-template-columns: 1fr; }
  .landing .feature {
    border-left: none;
    border-top: 1px solid var(--vp-c-divider);
  }
  .landing .feature:first-child { border-top: none; }
}
</style>
