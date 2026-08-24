import React, { useRef, useEffect, useState, useCallback } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import gsap from 'gsap'
import { CAMERA_PRESETS, PRESENTATION_VIEWS } from '../config/spaceConfig'

/**
 * 镜头控制系统 v2
 * OrbitControls + 6个预设镜头 + 6个展示模式镜头 + 自动巡游模式
 */
export default function CameraController({
  uiState,
  setUiState,
  isPresentationMode,
  activeViewIndex,
}) {
  const { camera } = useThree()
  const controlsRef = useRef()
  const tourIntervalRef = useRef(null)
  const [tourIndex, setTourIndex] = useState(0)

  const presetList = Object.entries(CAMERA_PRESETS)
  const presentationList = Object.entries(PRESENTATION_VIEWS)

  // === 镜头平滑移动 ===
  const moveCameraTo = useCallback((position, target, duration = 2) => {
    if (!controlsRef.current) return

    const currentPos = {
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
    }
    const currentTarget = {
      x: controlsRef.current.target.x,
      y: controlsRef.current.target.y,
      z: controlsRef.current.target.z,
    }

    gsap.to(currentPos, {
      x: position[0],
      y: position[1],
      z: position[2],
      duration,
      ease: 'power2.inOut',
      onUpdate: () => {
        camera.position.set(currentPos.x, currentPos.y, currentPos.z)
      },
    })

    gsap.to(currentTarget, {
      x: target[0],
      y: target[1],
      z: target[2],
      duration,
      ease: 'power2.inOut',
      onUpdate: () => {
        controlsRef.current.target.set(currentTarget.x, currentTarget.y, currentTarget.z)
      },
    })
  }, [camera])

  // === 普通模式: 切换预设镜头 ===
  useEffect(() => {
    if (!isPresentationMode && uiState.activePreset !== null) {
      const preset = presetList[uiState.activePreset]?.[1]
      if (preset) {
        const duration = uiState.isAutoTour ? 3.5 : 2
        moveCameraTo(preset.position, preset.target, duration)
      }
    }
  }, [isPresentationMode, uiState.activePreset, uiState.isAutoTour, moveCameraTo, presetList])

  // === 展示模式: 切换展示镜头 ===
  useEffect(() => {
    if (isPresentationMode && activeViewIndex !== undefined) {
      const view = presentationList[activeViewIndex]?.[1]
      if (view) {
        moveCameraTo(view.position, view.target, 1.8)
      }
    }
  }, [isPresentationMode, activeViewIndex, moveCameraTo, presentationList])

  // === 自动巡游模式 ===
  useEffect(() => {
    if (uiState.isAutoTour) {
      const STAY_DURATION = 5000
      if (uiState.activePreset === null) {
        setUiState(prev => ({ ...prev, activePreset: 0 }))
        setTourIndex(0)
      }

      tourIntervalRef.current = setInterval(() => {
        setTourIndex(prev => {
          const next = (prev + 1) % presetList.length
          setUiState(prevState => ({ ...prevState, activePreset: next }))
          return next
        })
      }, STAY_DURATION)

      return () => {
        if (tourIntervalRef.current) {
          clearInterval(tourIntervalRef.current)
          tourIntervalRef.current = null
        }
      }
    } else {
      if (tourIntervalRef.current) {
        clearInterval(tourIntervalRef.current)
        tourIntervalRef.current = null
      }
    }
  }, [uiState.isAutoTour, presetList.length, setUiState])

  // === 初始镜头 ===
  useEffect(() => {
    if (!isPresentationMode) {
      const preset = CAMERA_PRESETS.ENTRANCE_VIEW
      camera.position.set(...preset.position)
      if (controlsRef.current) {
        controlsRef.current.target.set(...preset.target)
      }
    }
  }, [camera, isPresentationMode])

  // 每帧更新controls
  useFrame(() => {
    if (controlsRef.current) {
      controlsRef.current.update()
    }
  })

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping
      dampingFactor={0.08}
      rotateSpeed={isPresentationMode ? 0.3 : 0.5}
      zoomSpeed={isPresentationMode ? 0.5 : 0.8}
      panSpeed={isPresentationMode ? 0.3 : 0.6}
      minDistance={1.5}
      maxDistance={30}
      maxPolarAngle={Math.PI / 2.05}
      minPolarAngle={0.05}
      enablePan={true}
      onStart={() => {
        if (!uiState.isAutoTour) {
          setUiState(prev => ({ ...prev, activePreset: null }))
        }
      }}
    />
  )
}
