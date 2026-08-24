import React from 'react'

/**
 * 顶部信息栏
 * 项目名称 + 策展主题简介
 */
export default function TopBar() {
  return (
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        padding: '20px 28px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        pointerEvents: 'none',
      }}
    >
      {/* 左侧：项目标题 */}
      <div
        style={{
          background: 'rgba(15, 30, 40, 0.6)',
          backdropFilter: 'blur(10px)',
          WebkitBackdropFilter: 'blur(10px)',
          borderRadius: '12px',
          padding: '14px 20px',
          border: '1px solid rgba(168, 208, 200, 0.2)',
          pointerEvents: 'auto',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #7fb3a5, #a8d0c8)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '16px',
              fontWeight: 700,
              color: '#0a1a18',
              letterSpacing: '0.05em',
            }}
          >
            浙
          </div>
          <div>
            <div
              style={{
                color: '#e8f0ed',
                fontSize: '16px',
                fontWeight: 700,
                letterSpacing: '0.25em',
              }}
            >
              浙韵焕新
            </div>
            <div
              style={{
                color: '#7fb3a5',
                fontSize: '10px',
                letterSpacing: '0.3em',
                marginTop: '2px',
              }}
            >
              ZHEJIANG PAVILION
            </div>
          </div>
        </div>
      </div>

      {/* 右侧：策展主题 */}
      <div
        style={{
          background: 'rgba(15, 30, 40, 0.6)',
          backdropFilter: 'blur(10px)',
          WebkitBackdropFilter: 'blur(10px)',
          borderRadius: '12px',
          padding: '14px 20px',
          border: '1px solid rgba(168, 208, 200, 0.2)',
          pointerEvents: 'auto',
          maxWidth: '520px',
        }}
      >
        <div
          style={{
            color: '#a8d0c8',
            fontSize: '11px',
            letterSpacing: '0.2em',
            marginBottom: '6px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}
        >
          <span style={{ color: '#7fb3a5' }}>◆</span>
          策展主题
        </div>
        <div
          style={{
            color: '#d0dcda',
            fontSize: '12px',
            lineHeight: '1.7',
            letterSpacing: '0.04em',
          }}
        >
          浙江历史经典产业通过
          <span style={{ color: '#7fb3a5', fontWeight: 600 }}> 文化传承 </span>·
          <span style={{ color: '#5a8cb5', fontWeight: 600 }}> 科技创新 </span>·
          <span style={{ color: '#6bb5a0', fontWeight: 600 }}> 设计赋能 </span>·
          <span style={{ color: '#c4b89a', fontWeight: 600 }}> 品牌重塑</span>
          <br />
          实现产业升级并形成
          <span
            style={{
              color: '#a8d0c8',
              fontWeight: 700,
              textShadow: '0 0 12px rgba(168,208,200,0.4)',
            }}
          >
            新质生产力
          </span>
        </div>
      </div>
    </div>
  )
}
