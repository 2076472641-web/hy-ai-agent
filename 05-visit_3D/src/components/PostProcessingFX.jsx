import React from 'react'
import { EffectComposer, Bloom, DepthOfField, Vignette, ChromaticAberration } from '@react-three/postprocessing'
import { BlendFunction } from 'postprocessing'

/**
 * 后处理特效系统 v2
 * 电影级画面 - Bloom辉光 + 景深 + 暗角 + 色差
 * 展示模式下参数更专业
 */
export default function PostProcessingFX({ isPresentationMode }) {
  if (isPresentationMode) {
    return (
      <EffectComposer multisampling={4} disableNormalPass>
        <Bloom
          intensity={0.7}
          luminanceThreshold={0.35}
          luminanceSmoothing={0.9}
          mipmapBlur
        />
        <DepthOfField
          focusDistance={0.012}
          focalLength={0.025}
          bokehScale={1.8}
        />
        <ChromaticAberration
          offset={[0.0006, 0.001]}
          radialModulation={false}
          modulationOffset={0}
          blendFunction={BlendFunction.NORMAL}
        />
        <Vignette
          offset={0.4}
          darkness={0.55}
          eskil={false}
        />
      </EffectComposer>
    )
  }

  return (
    <EffectComposer multisampling={0} disableNormalPass>
      <Bloom
        intensity={0.9}
        luminanceThreshold={0.2}
        luminanceSmoothing={0.85}
        mipmapBlur
      />
      <DepthOfField
        focusDistance={0.015}
        focalLength={0.022}
        bokehScale={2.2}
      />
      <ChromaticAberration
        offset={[0.0008, 0.0012]}
        radialModulation={false}
        modulationOffset={0}
        blendFunction={BlendFunction.NORMAL}
      />
      <Vignette
        offset={0.35}
        darkness={0.65}
        eskil={false}
      />
    </EffectComposer>
  )
}
