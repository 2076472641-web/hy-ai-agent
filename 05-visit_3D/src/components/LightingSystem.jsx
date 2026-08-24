import React from 'react'
import { COLORS, SPACE, ZONES } from '../config/spaceConfig'

/**
 * 专业灯光系统 v3 - 三层布光
 * 基础环境光 + 中央装置主光 + 四大展区洗墙 + 氛围点光源
 * 轻量化 (无rectAreaLight),兼顾FPS与层次
 */
export default function LightingSystem() {
  const { HALF_DEPTH, HALF_WIDTH, HEIGHT } = SPACE

  return (
    <group name="lighting-system">
      {/* === 环境光 - 整体基础照明 === */}
      <ambientLight color="#e8f0f2" intensity={0.55} />

      {/* === 半球光 - 天空/地面反射 === */}
      <hemisphereLight args={['#d4e4ec', '#1a2832', 0.5]} />

      {/* === 01 中央装置主射灯 (核心焦点) === */}
      <spotLight
        position={[0, HEIGHT - 0.6, 0]}
        angle={Math.PI / 5}
        penumbra={0.6}
        intensity={2.2}
        color="#f5faf6"
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
        shadow-bias={-0.0001}
        target-position={[0, 2.0, 0]}
      />
      {/* 中央辅助补光 - 稍低角度 */}
      <spotLight
        position={[3, 4, 3]}
        angle={Math.PI / 6}
        penumbra={0.8}
        intensity={0.9}
        color="#d8ecdf"
        target-position={[0, 1.8, 0]}
      />

      {/* === 02 入口门头射灯 === */}
      <spotLight
        position={[0, HEIGHT - 0.5, -HALF_DEPTH + 1.5]}
        angle={Math.PI / 4}
        penumbra={0.6}
        intensity={1.8}
        color="#f5faf5"
        target-position={[0, 3.5, -HALF_DEPTH + 0.3]}
      />
      {/* 入口地面洗光 */}
      <spotLight
        position={[0, 3.5, -HALF_DEPTH + 1]}
        angle={Math.PI / 3}
        penumbra={0.7}
        intensity={1.2}
        color="#eaf3ef"
        target-position={[0, 0.1, -HALF_DEPTH + 1.5]}
      />

      {/* === 03 四大展区洗墙光 (中层展墙) === */}
      {/* 源区 */}
      <spotLight
        position={[7.5, HEIGHT - 1.2, -3]}
        angle={Math.PI / 4}
        penumbra={0.7}
        intensity={1.3}
        color="#f0f7f2"
        target-position={[6.5, 1.4, -4]}
      />
      {/* 涌区 */}
      <spotLight
        position={[-7.5, HEIGHT - 1.2, -3]}
        angle={Math.PI / 4}
        penumbra={0.7}
        intensity={1.3}
        color="#edf3f7"
        target-position={[-6.5, 1.4, -4]}
      />
      {/* 涟区 */}
      <spotLight
        position={[-7.5, HEIGHT - 1.2, 3]}
        angle={Math.PI / 4}
        penumbra={0.7}
        intensity={1.2}
        color="#eaf4ef"
        target-position={[-6.5, 1.4, 4]}
      />
      {/* 汇区 */}
      <spotLight
        position={[7.5, HEIGHT - 1.2, 3]}
        angle={Math.PI / 4}
        penumbra={0.7}
        intensity={1.2}
        color="#edf1f7"
        target-position={[6.5, 1.4, 4]}
      />
      {/* 润区 (后方) */}
      <spotLight
        position={[0, HEIGHT - 1, 5.5]}
        angle={Math.PI / 3.5}
        penumbra={0.8}
        intensity={1.3}
        color="#f7f2ea"
        target-position={[0, 1.4, 5.8]}
      />

      {/* === 04 低层展柜补光 (暖色定向) === */}
      {/* 源/涌 展柜方向 */}
      <pointLight
        position={[-4, 1.2, -3]}
        color="#fff5dc"
        intensity={0.7}
        distance={4}
        decay={2}
      />
      <pointLight
        position={[4, 1.2, -3]}
        color="#fff5dc"
        intensity={0.7}
        distance={4}
        decay={2}
      />

      {/* === 05 氛围光 === */}
      {/* 中央装置青瓷氛围 */}
      <pointLight
        position={[0, 2.2, 0]}
        color={COLORS.CELADON_GREEN}
        intensity={1.2}
        distance={8}
        decay={1.8}
      />
      {/* 润区暖光 */}
      <pointLight
        position={[0, 2.5, 5.5]}
        color={COLORS.WARM_LIGHT}
        intensity={1.1}
        distance={6}
        decay={2}
      />
      {/* 入口欢迎光 */}
      <pointLight
        position={[0, 0.5, -HALF_DEPTH + 1.8]}
        color={COLORS.LIGHT_BLUE_GREEN}
        intensity={0.9}
        distance={5}
        decay={2}
      />
    </group>
  )
}
