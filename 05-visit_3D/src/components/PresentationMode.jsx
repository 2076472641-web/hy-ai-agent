import React, { useRef, useCallback, useState } from 'react'
import { PRESENTATION_VIEWS } from '../config/spaceConfig'

const viewKeys = Object.keys(PRESENTATION_VIEWS)

/**
 * Presentation Mode - 专业展示模式
 * 6个建筑摄影镜头 + Capture高清截图
 */
export default function PresentationMode({
  isPresentationMode,
  setIsPresentationMode,
  activeViewIndex,
  setActiveViewIndex,
  onCapture,
  captureResolution,
  setCaptureResolution,
}) {
  const [isCapturing, setIsCapturing] = useState(false)

  const currentView = viewKeys[activeViewIndex]
  const viewData = currentView ? PRESENTATION_VIEWS[currentView] : null

  const handleCapture = useCallback(() => {
    if (onCapture) {
      setIsCapturing(true)
      setTimeout(() => {
        onCapture(captureResolution)
        setIsCapturing(false)
      }, 100)
    }
  }, [onCapture, captureResolution])

  return (
    <div
      style={{
        position: 'absolute',
        right: '24px',
        bottom: '24px',
        zIndex: 100,
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        pointerEvents: 'auto',
        minWidth: '280px',
      }}
    >
      {/* === 展示模式切换 === */}
      <div
        style={{
          background: 'rgba(15, 30, 40, 0.85)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          borderRadius: '14px',
          padding: '14px',
          border: '1px solid rgba(200, 180, 120, 0.4)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '10px',
            paddingBottom: '8px',
            borderBottom: '1px solid rgba(200, 180, 120, 0.25)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div
              style={{
                width: '4px',
                height: '16px',
                background: 'linear-gradient(to bottom, #c4b89a, #e8dcc0)',
                borderRadius: '2px',
              }}
            />
            <span
              style={{
                color: '#e8dcc0',
                fontSize: '13px',
                fontWeight: 600,
                letterSpacing: '0.15em',
              }}
            >
              展示模式
            </span>
          </div>
          <button
            onClick={() => setIsPresentationMode(!isPresentationMode)}
            style={{
              padding: '4px 10px',
              borderRadius: '6px',
              border: isPresentationMode
                ? '1px solid rgba(200, 180, 120, 0.8)'
                : '1px solid rgba(255,255,255,0.15)',
              background: isPresentationMode
                ? 'linear-gradient(90deg, rgba(200,180,120,0.25), rgba(232,220,192,0.1))'
                : 'rgba(255,255,255,0.04)',
              color: isPresentationMode ? '#f0e8d4' : '#809098',
              cursor: 'pointer',
              fontSize: '11px',
              fontFamily: 'inherit',
              letterSpacing: '0.1em',
              transition: 'all 0.2s',
            }}
          >
            {isPresentationMode ? 'ON' : 'OFF'}
          </button>
        </div>

        {isPresentationMode && (
          <>
            {/* === 6个建筑摄影镜头 === */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
              {viewKeys.map((key, i) => {
                const view = PRESENTATION_VIEWS[key]
                const isActive = activeViewIndex === i
                return (
                  <button
                    key={key}
                    onClick={() => setActiveViewIndex(i)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '10px',
                      width: '100%',
                      padding: '9px 10px',
                      borderRadius: '8px',
                      border: isActive
                        ? '1px solid rgba(200, 180, 120, 0.8)'
                        : '1px solid rgba(255,255,255,0.06)',
                      background: isActive
                        ? 'linear-gradient(90deg, rgba(200,180,120,0.2), rgba(232,220,192,0.08))'
                        : 'rgba(255,255,255,0.02)',
                      color: isActive ? '#f0e8d4' : '#809098',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      textAlign: 'left',
                      fontFamily: 'inherit',
                      fontSize: '12px',
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = 'rgba(255,255,255,0.06)'
                        e.currentTarget.style.color = '#c0c8cc'
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.background = 'rgba(255,255,255,0.02)'
                        e.currentTarget.style.color = '#809098'
                      }
                    }}
                  >
                    <span
                      style={{
                        width: '32px',
                        height: '20px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '10px',
                        fontWeight: 700,
                        color: isActive ? '#c4b89a' : '#5a6870',
                        background: isActive
                          ? 'rgba(200,180,120,0.15)'
                          : 'rgba(255,255,255,0.04)',
                        borderRadius: '3px',
                        letterSpacing: '0.05em',
                      }}
                    >
                      {view.name.split(' ')[0]}
                    </span>
                    <span style={{ flex: 1, letterSpacing: '0.04em' }}>
                      {view.name.split(' ')[1]}
                    </span>
                    {isActive && (
                      <div
                        style={{
                          width: '5px',
                          height: '5px',
                          borderRadius: '50%',
                          background: '#c4b89a',
                          boxShadow: '0 0 6px #c4b89a',
                        }}
                      />
                    )}
                  </button>
                )
              })}
            </div>

            {/* === 分辨率选择 === */}
            <div
              style={{
                marginTop: '10px',
                padding: '10px',
                borderRadius: '8px',
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.05)',
              }}
            >
              <div
                style={{
                  fontSize: '10px',
                  color: '#7a8a90',
                  letterSpacing: '0.15em',
                  marginBottom: '8px',
                }}
              >
                输出分辨率
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                {[
                  { value: '1920x1080', label: 'FHD' },
                  { value: '2560x1440', label: '2K' },
                ].map((res) => (
                  <button
                    key={res.value}
                    onClick={() => setCaptureResolution(res.value)}
                    style={{
                      flex: 1,
                      padding: '6px 8px',
                      borderRadius: '6px',
                      border:
                        captureResolution === res.value
                          ? '1px solid rgba(200, 180, 120, 0.7)'
                          : '1px solid rgba(255,255,255,0.08)',
                      background:
                        captureResolution === res.value
                          ? 'rgba(200,180,120,0.2)'
                          : 'rgba(255,255,255,0.03)',
                      color:
                        captureResolution === res.value
                          ? '#f0e8d4'
                          : '#7a8a90',
                      cursor: 'pointer',
                      fontFamily: 'inherit',
                      fontSize: '11px',
                      letterSpacing: '0.05em',
                      transition: 'all 0.2s',
                    }}
                  >
                    {res.label}
                  </button>
                ))}
              </div>
            </div>

            {/* === Capture 按钮 === */}
            <button
              onClick={handleCapture}
              disabled={isCapturing}
              style={{
                marginTop: '10px',
                width: '100%',
                padding: '12px 14px',
                borderRadius: '10px',
                border: 'none',
                cursor: isCapturing ? 'not-allowed' : 'pointer',
                fontFamily: 'inherit',
                fontSize: '13px',
                fontWeight: 700,
                letterSpacing: '0.2em',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '10px',
                transition: 'all 0.25s ease',
                background: isCapturing
                  ? 'linear-gradient(135deg, #4a4030, #6a5840)'
                  : 'linear-gradient(135deg, #c4b89a, #a89868)',
                color: isCapturing ? '#809098' : '#1a1408',
                boxShadow: isCapturing
                  ? 'none'
                  : '0 4px 20px rgba(200,180,120,0.35)',
              }}
            >
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: isCapturing ? '#6a7078' : '#1a1408',
                  boxShadow: isCapturing ? 'none' : '0 0 8px rgba(0,0,0,0.4)',
                }}
              />
              {isCapturing ? 'CAPTURING...' : 'CAPTURE'}
            </button>
          </>
        )}
      </div>
    </div>
  )
}
