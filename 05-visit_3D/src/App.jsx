import React, { useState, useCallback, useEffect, useRef } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { Stars } from '@react-three/drei'
import * as THREE from 'three'

// 3D场景组件
import PavilionShell from './components/PavilionShell'
import EntranceGate from './components/EntranceGate'
import CoreArtInstallation from './components/CoreArtInstallation'
import Zones from './components/Zones'
import HumanFigures from './components/HumanFigures'
import LightingSystem from './components/LightingSystem'
import CameraController from './components/CameraController'
import PostProcessingFX from './components/PostProcessingFX'
import DebugMode from './components/DebugMode'

// UI组件
import TopBar from './components/TopBar'
import ControlPanel from './components/ControlPanel'
import PresentationMode from './components/PresentationMode'

import { COLORS, PRESENTATION_VIEWS } from './config/spaceConfig'

// 相机同步组件 - 处理展示模式下的FOV切换
function CameraSync({ isPresentationMode, activeViewIndex }) {
  const { camera } = useThree()

  useEffect(() => {
    if (isPresentationMode) {
      const viewKeys = Object.keys(PRESENTATION_VIEWS)
      const viewKey = viewKeys[activeViewIndex]
      const view = PRESENTATION_VIEWS[viewKey]
      if (view && view.fov) {
        camera.fov = view.fov
        camera.updateProjectionMatrix()
      }
    } else {
      camera.fov = 50
      camera.updateProjectionMatrix()
    }
  }, [isPresentationMode, activeViewIndex, camera])

  return null
}

// 截图桥接组件 - 在Canvas内部捕获renderer/scene/camera引用
function CaptureBridge({ rendererRef, sceneRef, cameraRef }) {
  const { gl, scene, camera } = useThree()

  useEffect(() => {
    rendererRef.current = gl
    sceneRef.current = scene
    cameraRef.current = camera
  }, [gl, scene, camera, rendererRef, sceneRef, cameraRef])

  return null
}

export default function App() {
  // UI状态
  const [uiState, setUiState] = useState({
    activePreset: null,
    isAutoTour: false,
  })

  // 展示模式状态
  const [isPresentationMode, setIsPresentationMode] = useState(false)
  const [activeViewIndex, setActiveViewIndex] = useState(0)
  const [captureResolution, setCaptureResolution] = useState('1920x1080')
  const [debugMode, setDebugMode] = useState(false)

  // 存储Three.js渲染器/场景/相机引用
  const glRendererRef = useRef(null)
  const glSceneRef = useRef(null)
  const glCameraRef = useRef(null)

  // 截图处理 - 使用renderer.setSize + canvas.toDataURL方案
  const handleCapture = useCallback(
    (resolution) => {
      const [w, h] = resolution.split('x').map(Number)
      const canvasEl = document.querySelector('canvas')
      if (!canvasEl) return

      // 创建全屏覆盖层隐藏所有UI
      const captureOverlay = document.createElement('div')
      captureOverlay.style.cssText = 'position:fixed;inset:0;background:transparent;z-index:9999;pointer-events:none;'
      document.body.appendChild(captureOverlay)

      // 隐藏所有面板UI
      const panels = document.querySelectorAll('div[style*="position: absolute"]')
      const hiddenPanels = []
      panels.forEach(p => {
        const zIndex = parseInt(p.style.zIndex)
        if (zIndex === 100 || zIndex === 90) {
          hiddenPanels.push({ el: p, display: p.style.display })
          p.style.display = 'none'
        }
      })

      // 等待画面稳定后截屏
      setTimeout(() => {
        try {
          const renderer = glRendererRef.current
          const scene = glSceneRef.current
          const camera = glCameraRef.current

          if (!renderer || !scene || !camera) {
            throw new Error('Three.js renderer/scene/camera not available')
          }

          const originalW = canvasEl.width
          const originalH = canvasEl.height

          // 保存当前canvas内容到临时canvas
          const tmpCanvas = document.createElement('canvas')
          tmpCanvas.width = originalW
          tmpCanvas.height = originalH
          const tmpCtx = tmpCanvas.getContext('2d')
          tmpCtx.drawImage(canvasEl, 0, 0)

          // 通过renderer.setSize调整分辨率（不改变CSS尺寸，只改变绘制缓冲）
          renderer.setSize(w, h, false)

          // 重新渲染场景（EffectComposer会自动处理后处理）
          renderer.render(scene, camera)

          // 立即使用canvas.toDataURL导出（preserveDrawingBuffer在渲染后短时间内有效）
          const link = document.createElement('a')
          const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
          link.download = `zhejiang-pavilion-${w}x${h}-${timestamp}.png`
          link.href = canvasEl.toDataURL('image/png')
          link.click()

          // 恢复原始分辨率
          renderer.setSize(originalW, originalH, false)

        } catch (err) {
          console.error('Capture failed:', err)
          // Fallback: 直接导出当前canvas
          const canvasEl2 = document.querySelector('canvas')
          if (canvasEl2) {
            const link = document.createElement('a')
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
            link.download = `zhejiang-pavilion-${resolution}-${timestamp}.png`
            link.href = canvasEl2.toDataURL('image/png')
            link.click()
          }
        }

        // 恢复UI
        setTimeout(() => {
          hiddenPanels.forEach(({ el, display }) => {
            el.style.display = display
          })
          if (document.body.contains(captureOverlay)) {
            document.body.removeChild(captureOverlay)
          }
        }, 1500)
      }, 500)
    },
    []
  )

  const isUIHidden = isPresentationMode

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: COLORS.DARK_BG }}>
      {/* ============================================
          HTML层 - 可隐藏UI覆盖层
      ============================================ */}
      {!isUIHidden && <TopBar />}
      {!isUIHidden && <ControlPanel uiState={uiState} setUiState={setUiState} />}

      {/* ============================================
          展示模式 UI (右侧)
      ============================================ */}
      <PresentationMode
        isPresentationMode={isPresentationMode}
        setIsPresentationMode={setIsPresentationMode}
        activeViewIndex={activeViewIndex}
        setActiveViewIndex={setActiveViewIndex}
        onCapture={handleCapture}
        captureResolution={captureResolution}
        setCaptureResolution={setCaptureResolution}
      />

      {/* ============================================
          展示模式提示
      ============================================ */}
      {isPresentationMode && (
        <div
          style={{
            position: 'absolute',
            top: '20px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 90,
            padding: '10px 20px',
            background: 'rgba(15, 30, 40, 0.75)',
            backdropFilter: 'blur(10px)',
            borderRadius: '10px',
            border: '1px solid rgba(200, 180, 120, 0.3)',
            color: '#e8dcc0',
            fontSize: '12px',
            letterSpacing: '0.2em',
            pointerEvents: 'none',
            fontWeight: 600,
          }}
        >
          浙韵焕新 · 专业展示模式
        </div>
      )}

      {/* ============================================
          Debug Mode 切换按钮 (右下角)
      ============================================ */}
      {!isUIHidden && (
        <button
          onClick={() => setDebugMode(!debugMode)}
          style={{
            position: 'absolute',
            right: '24px',
            top: '80px',
            zIndex: 100,
            padding: '8px 14px',
            borderRadius: '8px',
            border: debugMode
              ? '1px solid rgba(255, 100, 100, 0.7)'
              : '1px solid rgba(255,255,255,0.1)',
            background: debugMode
              ? 'rgba(200, 50, 50, 0.25)'
              : 'rgba(15, 30, 40, 0.7)',
            backdropFilter: 'blur(8px)',
            color: debugMode ? '#ff8888' : '#7a8a90',
            cursor: 'pointer',
            fontFamily: 'inherit',
            fontSize: '11px',
            fontWeight: 600,
            letterSpacing: '0.1em',
            transition: 'all 0.2s',
          }}
        >
          {debugMode ? 'DEBUG ON' : 'DEBUG'}
        </button>
      )}

      {/* ============================================
          3D Canvas 场景
      ============================================ */}
      <Canvas
        shadows
        camera={{
          fov: 50,
          near: 0.1,
          far: 200,
          position: [0, 1.7, -10],
        }}
        gl={{
          antialias: true,
          alpha: false,
          powerPreference: 'high-performance',
          toneMapping: 3,  // ACESFilmicToneMapping
          toneMappingExposure: 1.2,
          stencil: false,
          shadowMapType: THREE.PCFSoftShadowMap,
          preserveDrawingBuffer: true,  // 保持绘制缓冲用于截图导出
        }}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
        }}
        dpr={[1, 2]}
        frameloop="always"
      >
        {/* === 截图桥接 - 捕获renderer/scene/camera === */}
        <CaptureBridge
          rendererRef={glRendererRef}
          sceneRef={glSceneRef}
          cameraRef={glCameraRef}
        />

        {/* === 背景色 === */}
        <color attach="background" args={[COLORS.DARK_BG]} />
        {/* 雾效 - 增强空间深度感 */}
        <fog attach="fog" args={[COLORS.DARK_BG, 20, 60]} />

        {/* === 远景微量粒子 - 增强数字科技氛围 === */}
        <Stars
          radius={100}
          depth={30}
          count={300}
          factor={2}
          saturation={0.2}
          fade
          speed={0.2}
          color={COLORS.LIGHT_BLUE_GREEN}
        />

        {/* === 镜头控制系统 === */}
        <CameraController
          uiState={uiState}
          setUiState={setUiState}
          isPresentationMode={isPresentationMode}
          activeViewIndex={activeViewIndex}
        />

        {/* === 相机同步 - 展示模式 FOV 切换 === */}
        <CameraSync
          isPresentationMode={isPresentationMode}
          activeViewIndex={activeViewIndex}
        />

        {/* === 专业灯光系统 === */}
        <LightingSystem />

        {/* === 展馆外壳结构 === */}
        <PavilionShell />

        {/* === 入口门头 === */}
        <EntranceGate />

        {/* === 中央艺术装置 - 潮生之芯 === */}
        <CoreArtInstallation />

        {/* === 五大主题展区 === */}
        <Zones />

        {/* === 人体尺度参照 === */}
        <HumanFigures />

        {/* === Debug Mode 空间验证 === */}
        <DebugMode visible={debugMode} />

        {/* === 后处理特效 - 电影级画面 === */}
        <PostProcessingFX isPresentationMode={isPresentationMode} />
      </Canvas>
    </div>
  )
}
