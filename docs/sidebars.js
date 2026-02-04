// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */

const sidebars = {
  tutorialSidebar: [
    'intro',
    'quickstart',

    {
      type: 'category',
      label: 'Guides',
      link: { type: 'doc', id: 'guides/index' },
      collapsed: false,
      items: [
        'guides/data-acquisition',
        'guides/synchronous-mode',
        'guides/asynchronous-mode',
        'guides/neurofeedback-mode',
        'guides/send-events',
      ],
    },

    {
      type: 'category',
      label: 'System Architecture',
      link: { type: 'doc', id: 'conceptual/conceptual-overview' },
      collapsed: true,
      items: [
        'conceptual/architecture/task-application-layer',
        'conceptual/architecture/data-layer',
        'conceptual/architecture/processing-layer',
        'conceptual/architecture/ml-layer',
        'conceptual/architecture/auxiliary-layer',
      ],
    },

    {
      type: 'category',
      label: 'API Reference',
      link: { type: 'doc', id: 'api/api-overview' },
      collapsed: true,
      items: [
        {
          type: 'category',
          label: 'dendrite.data',
          items: [
            'api/generated/dendrite/data/event_outlet',
          ],
        },
        {
          type: 'category',
          label: 'dendrite.ml.decoders',
          items: [
            'api/generated/dendrite/ml/decoders/decoders',
            'api/generated/dendrite/ml/decoders/decoder',
            'api/generated/dendrite/ml/decoders/decoder_schemas',
            'api/generated/dendrite/ml/decoders/registry',
          ],
        },
        {
          type: 'category',
          label: 'dendrite.ml.models',
          items: [
            'api/generated/dendrite/ml/models/models',
            'api/generated/dendrite/ml/models/base',
          ],
        },
      ],
    },

    {
      type: 'category',
      label: 'Resources',
      collapsed: true,
      items: [
        'changelog',
      ],
    },
  ],
};

export default sidebars;
