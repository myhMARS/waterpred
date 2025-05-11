// utils/message.js
import {createVNode, render} from 'vue'
import Notification from '@/components/ui/notification/Notification.vue'

let container = null

function getContainer() {
  if (!container) {
    container = document.createElement('div')
    container.className = 'fixed top-4 right-4 space-y-2 z-50'
    document.body.appendChild(container)
  }
  return container
}

export function showMessage({ type = 'info', message = '', duration = 2000, onClose = () => {} }) {
  const wrapper = document.createElement('div')
  const parent = getContainer()
  parent.appendChild(wrapper)

  const close = () => {
    render(null, wrapper)
    parent.removeChild(wrapper)
    onClose()
  }

  const vnode = createVNode(Notification, {
    type,
    message,
    modelValue: true,
    'onUpdate:modelValue': (val) => {
      if (!val) close()
    },
  })

  render(vnode, wrapper)

  if (duration > 0) {
    setTimeout(() => {
      close()
    }, duration)
  }
}
