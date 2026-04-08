import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'control',
      component: () => import('../views/ControlView.vue'),
    },
    {
      path: '/data',
      name: 'data',
      component: () => import('../views/DataExplorerView.vue'),
    },
    {
      path: '/ml',
      name: 'ml',
      component: () => import('../views/MLWorkbenchView.vue'),
    },
  ],
})

export default router
