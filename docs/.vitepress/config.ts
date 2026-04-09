import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/dendrite/',
  title: 'Dendrite',
  description: 'Open-source platform for multimodal signal acquisition, real-time processing, and decoder training via LSL',

  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Quickstart', link: '/quickstart' },
      { text: 'Guides', link: '/guides/' },
      { text: 'Architecture', link: '/architecture/' },
      { text: 'API', link: '/api' },
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Quickstart', link: '/quickstart' },
        ],
      },
      {
        text: 'Guides',
        items: [
          { text: 'Core Concepts', link: '/guides/concepts' },
          { text: 'Data Acquisition', link: '/guides/data-acquisition' },
          { text: 'Synchronous Mode', link: '/guides/synchronous-mode' },
          { text: 'Asynchronous Mode', link: '/guides/asynchronous-mode' },
          { text: 'Neurofeedback Mode', link: '/guides/neurofeedback-mode' },
          { text: 'ML Training', link: '/guides/ml-training' },
          { text: 'Stream Replay', link: '/guides/stream-replay' },
          { text: 'Send Events', link: '/guides/send-events' },
        ],
      },
      {
        text: 'Architecture',
        items: [
          { text: 'Overview', link: '/architecture/' },
          { text: 'Web Layer', link: '/architecture/web-layer' },
          { text: 'Data Layer', link: '/architecture/data-layer' },
          { text: 'Processing Layer', link: '/architecture/processing-layer' },
          { text: 'ML Layer', link: '/architecture/ml-layer' },
          { text: 'Task Application Layer', link: '/architecture/task-application-layer' },
        ],
      },
      {
        text: 'Reference',
        items: [
          { text: 'API Reference', link: '/api' },
          { text: 'Changelog', link: '/changelog' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/dendrite-bci/dendrite' },
    ],

    search: {
      provider: 'local',
    },
  },
})
