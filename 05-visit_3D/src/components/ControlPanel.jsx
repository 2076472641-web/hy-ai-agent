import React from 'react'
import { CAMERA_PRESETS } from '../config/spaceConfig'

const presetKeys = Object.keys(CAMERA_PRESETS)

/**
 * UI控制面板
 * 预设镜头切换 + 自动巡游开关
 */
export default function ControlPanel({ uiState, setUiState }) {
  const { activePreset, isAutoTour } = uiState

  const handlePresetClick = (index) => {
    // 如果正在自动巡游，先停止
    if (isAutoTour) {
      setUiState(prev => ({ ...prev, isAutoTour: false, activePreset: index }))
    } else {
      setUiState(prev => ({
        ...prev,
        activePreset: prev.activePreset === index ? null : index,
      }))
    }
  }

  const toggleAutoTour = () => {
    setUiState(prev => ({
      ...prev,
      isAutoTour: !prev.isAutoTour,
      activePreset: !prev.isAutoTour ? 0 : prev.activePreset,
    }))
  }

  return (
    <div
      style={{
        position: 'absolute',
        left: '24px',
        bottom: '24px',
        zIndex: 100,
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        pointerEvents: 'auto',
      }}
    >
      {/* === 镜头预设面板 === */}
      <div
        style={{
          background: 'rgba(15, 30, 40, 0.85)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          borderRadius: '14px',
          padding: '16px',
          border: '1px solid rgba(127, 179, 165, 0.3)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
          minWidth: '230px',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginBottom: '12px',
            paddingBottom: '10px',
            borderBottom: '1px solid rgba(168, 208, 200, 0.2)',
          }}
        >
          <div
            style={{
              width: '4px',
              height: '16px',
              background: 'linear-gradient(to bottom, #7fb3a5, #a8d0c8)',
              borderRadius: '2px',
            }}
          />
          <span
            style={{
              color: '#e8f0ed',
              fontSize: '13px',
              fontWeight: 600,
              letterSpacing: '0.1em',
            }}
          >
            镜头视角
          </span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {presetKeys.map((key, i) => {
            const preset = CAMERA_PRESETS[key]
            const isActive = activePreset === i
            const num = String(i + 1).padStart(2, '0')

            return (
              <button
                key={key}
                onClick={() => handlePresetClick(i)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  width: '100%',
                  padding: '10px 12px',
                  borderRadius: '8px',
                  border: isActive
                    ? '1px solid rgba(127, 179, 165, 0.8)'
                    : '1px solid rgba(255,255,255,0.08)',
                  background: isActive
                    ? 'linear-gradient(90deg, rgba(127,179,165,0.25), rgba(168,208,200,0.1))'
                    : 'rgba(255,255,255,0.03)',
                  color: isActive ? '#e8f5f0' : '#a8b5bb',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  textAlign: 'left',
                  fontFamily: 'inherit',
                  fontSize: '13px',
                }}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.07)'
                    e.currentTarget.style.color = '#d0dcda'
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.03)'
                    e.currentTarget.style.color = '#a8b5bb'
                  }
                }}
              >
                <span
                  style={{
                    width: '28px',
                    height: '22px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '11px',
                    fontWeight: 700,
                    color: isActive ? '#7fb3a5' : '#6a7880',
                    background: isActive
                      ? 'rgba(127,179,165,0.15)'
                      : 'rgba(255,255,255,0.04)',
                    borderRadius: '4px',
                    letterSpacing: '0.05em',
                  }}
                >
                  {num}
                </span>
                <span
                  style={{
                    flex: 1,
                    letterSpacing: '0.05em',
                  }}
                >
                  {preset.name}
                </span>
                {isActive && (
                  <div
                    style={{
                      width: '6px',
                      height: '6px',
                      borderRadius: '50%',
                      background: '#7fb3a5',
                      boxShadow: '0 0 8px #7fb3a5',
                    }}
                  />
                )}
              </button>
            )
          })}
        </div>

        {/* 自动巡游按钮 */}
        <button
          onClick={toggleAutoTour}
          style={{
            marginTop: '14px',
            width: '100%',
            padding: '12px 14px',
            borderRadius: '10px',
            border: 'none',
            cursor: 'pointer',
            fontFamily: 'inherit',
            fontSize: '13px',
            fontWeight: 600,
            letterSpacing: '0.15em',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
            transition: 'all 0.25s ease',
            background: isAutoTour
              ? 'linear-gradient(135deg, #7fb3a5, #a8d0c8)'
              : 'linear-gradient(135deg, rgba(127,179,165,0.2), rgba(168,208,200,0.1))',
            color: isAutoTour ? '#0a1a18' : '#c8d8d4',
            boxShadow: isAutoTour
              ? '0 4px 20px rgba(127,179,165,0.4)'
              : 'none',
          }}
        >
          <div
            style={{
              width: isAutoTour ? 0 : '0px',
              height: '0px',
              borderStyle: 'solid',
              borderWidth: isAutoTour ? '5px 0 5px 8px' : '0',
              borderColor: isAutoTour
                ? 'transparent transparent transparent #0a1a18'
                : 'transparent',
              transition: 'all 0.2s',
            }}
          />
          {isAutoTour ? (
            <div style={{ display: 'flex', gap: '3px' }}>
              <div
                style={{
                  width: '3px',
                  height: '12px',
                  background: '#0a1a18',
                  borderRadius: '1px',
                }}
              />
              <div
                style={{
                  width: '3px',
                  height: '12px',
                  background: '#0a1a18',
                  borderRadius: '1px',
                }}
              />
            </div>
          ) : null}
          <span>{isAutoTour ? '停止巡游' : '自动巡游'}</span>
        </button>
      </div>

      {/* === 操作提示 === */}
      <div
        style={{
          background: 'rgba(15, 30, 40, 0.7)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          borderRadius: '10px',
          padding: '10px 14px',
          border: '1px solid rgba(255,255,255,0.06)',
          fontSize: '11px',
          color: '#7a8a90',
          lineHeight: '1.7',
          letterSpacing: '0.03em',
        }}
      >
        <div>🖱️ 左键拖拽：旋转视角</div>
        <div>🖱️ 右键拖拽：平移视角</div>
        <div>🖱️ 滚轮：缩放远近</div>
      </div>
    </div>
  )
}
