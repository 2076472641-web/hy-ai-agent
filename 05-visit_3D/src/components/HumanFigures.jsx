import React from 'react'
import { COLORS } from '../config/spaceConfig'

/**
 * 人体尺度参照 - 1.7m 剪影
 * 5个人物分布在入口/中央/展区旁
 * 采用浅灰白色剪影+暖边光,在暗背景下清晰可见
 */
function HumanSilhouette({ position = [0, 0, 0], rotationY = 0, scale = 1 }) {
  return (
    <group position={position} rotation={[0, rotationY, 0]} scale={scale}>
      {/* 腿部 */}
      <mesh position={[-0.06, 0.3, 0]}>
        <boxGeometry args={[0.07, 0.55, 0.07]} />
        <meshStandardMaterial color="#d9d3c7" roughness={0.6} metalness={0.05} />
      </mesh>
      <mesh position={[0.06, 0.3, 0]}>
        <boxGeometry args={[0.07, 0.55, 0.07]} />
        <meshStandardMaterial color="#d9d3c7" roughness={0.6} metalness={0.05} />
      </mesh>
      {/* 身体 - 胶囊 */}
      <mesh position={[0, 0.95, 0]}>
        <capsuleGeometry args={[0.12, 0.55, 8, 16]} />
        <meshStandardMaterial color="#e8e2d5" roughness={0.55} metalness={0.05} />
      </mesh>
      {/* 头部 - 球 */}
      <mesh position={[0, 1.62, 0]}>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshStandardMaterial color="#efe9dd" roughness={0.5} metalness={0.05} />
      </mesh>
      {/* 胳膊 */}
      <mesh position={[-0.22, 1.0, 0]} rotation={[0.15, 0, 0.2]}>
        <boxGeometry args={[0.05, 0.45, 0.05]} />
        <meshStandardMaterial color="#d9d3c7" roughness={0.6} metalness={0.05} />
      </mesh>
      <mesh position={[0.22, 1.0, 0]} rotation={[0.15, 0, -0.2]}>
        <boxGeometry args={[0.05, 0.45, 0.05]} />
        <meshStandardMaterial color="#d9d3c7" roughness={0.6} metalness={0.05} />
      </mesh>
      {/* 轮廓发光 (增强在暗背景下识别) */}
      <pointLight color={COLORS.CELADON_GREEN} intensity={0.15} distance={0.8} decay={2} />
    </group>
  )
}

export default function HumanFigures() {
  return (
    <group name="human-figures">
      {/* 入口处 - 2人进入姿态 */}
      <HumanSilhouette position={[-1.8, 0, -4.5]} rotationY={0.3} />
      <HumanSilhouette position={[1.6, 0, -4.8]} rotationY={-0.4} />

      {/* 中央装置旁 - 1人观赏 */}
      <HumanSilhouette position={[2.8, 0, 1.8]} rotationY={-1.2} />

      {/* 源展区旁 - 1人驻足 */}
      <HumanSilhouette position={[5.2, 0, -2.2]} rotationY={-0.8} />

      {/* 涌展区旁 - 1人观看屏幕 */}
      <HumanSilhouette position={[-5.2, 0, -2.5]} rotationY={0.6} />

      {/* 后方通道 - 1人经过 */}
      <HumanSilhouette position={[0.8, 0, 5.2]} rotationY={3.1} />
    </group>
  )
}
