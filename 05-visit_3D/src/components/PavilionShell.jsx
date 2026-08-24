import React from 'react'
import { SPACE, COLORS } from '../config/spaceConfig'

/**
 * 展馆外壳 v2 - 专业临时展会结构
 * 21m × 13m × 6m 铝型材 + 软膜 + 模块化构件
 * 避免永久建筑感,强化临时展商结构逻辑
 */
export default function PavilionShell() {
  const { WIDTH, DEPTH, HEIGHT, HALF_WIDTH, HALF_DEPTH } = SPACE

  return (
    <group name="pavilion-shell">
      {/* === 地面 - 浅色PVC地胶 (专业展会常用) === */}
      <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <planeGeometry args={[WIDTH, DEPTH]} />
        <meshStandardMaterial
          color="#1d2b36"
          roughness={0.55}
          metalness={0.08}
        />
      </mesh>

      {/* 中央艺术装置基座 - 深色大理石质感 */}
      <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.004, 0]}>
        <circleGeometry args={[4.2, 64]} />
        <meshStandardMaterial
          color="#243c4c"
          roughness={0.3}
          metalness={0.25}
        />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.006, 0]}>
        <ringGeometry args={[4.2, 4.28, 64]} />
        <meshStandardMaterial
          color={COLORS.CELADON_GREEN}
          emissive={COLORS.CELADON_GREEN}
          emissiveIntensity={1.4}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* === 地面引导线 === */}
      {/* 纵向主通道引导线 (中线) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.008, 0]}>
        <planeGeometry args={[0.1, DEPTH - 2]} />
        <meshStandardMaterial
          color={COLORS.CELADON_GREEN}
          emissive={COLORS.CELADON_GREEN}
          emissiveIntensity={0.9}
          transparent
          opacity={0.75}
        />
      </mesh>
      {/* 横向引导线 (两条) */}
      {[-2.5, 2.5].map((z, i) => (
        <mesh key={`guide-${i}`} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.008, z]}>
          <planeGeometry args={[WIDTH - 4, 0.08]} />
          <meshStandardMaterial
            color={COLORS.LIGHT_BLUE_GREEN}
            emissive={COLORS.LIGHT_BLUE_GREEN}
            emissiveIntensity={0.6}
            transparent
            opacity={0.6}
          />
        </mesh>
      ))}
      {/* 分道线 - 辅助通道 */}
      {[-3.5, 3.5].map((x, i) => (
        <mesh key={`lane-${i}`} rotation={[-Math.PI / 2, 0, 0]} position={[x, 0.006, 0]}>
          <planeGeometry args={[0.03, DEPTH - 3]} />
          <meshStandardMaterial color={COLORS.LIGHT_BLUE_GREEN} emissive={COLORS.LIGHT_BLUE_GREEN} emissiveIntensity={0.35} transparent opacity={0.45} />
        </mesh>
      ))}

      {/* === 四角铝型材立柱 (100×100mm) === */}
      {[
        [-HALF_WIDTH + 0.5, -HALF_DEPTH + 0.5],
        [HALF_WIDTH - 0.5, -HALF_DEPTH + 0.5],
        [-HALF_WIDTH + 0.5, HALF_DEPTH - 0.5],
        [HALF_WIDTH - 0.5, HALF_DEPTH - 0.5],
      ].map(([x, z], i) => (
        <group key={`corner-${i}`}>
          <mesh castShadow position={[x, HEIGHT / 2, z]}>
            <boxGeometry args={[0.1, HEIGHT, 0.1]} />
            <meshStandardMaterial color={COLORS.SILVER_METAL} metalness={0.9} roughness={0.15} />
          </mesh>
          {/* 基座 */}
          <mesh position={[x, 0.05, z]}>
            <boxGeometry args={[0.35, 0.1, 0.35]} />
            <meshStandardMaterial color="#3a4852" roughness={0.6} metalness={0.3} />
          </mesh>
        </group>
      ))}

      {/* === 中部立柱 (支撑顶部横梁) === */}
      {[
        [-HALF_WIDTH + 0.5, -3],
        [HALF_WIDTH - 0.5, -3],
        [-HALF_WIDTH + 0.5, 0],
        [HALF_WIDTH - 0.5, 0],
        [-HALF_WIDTH + 0.5, 3],
        [HALF_WIDTH - 0.5, 3],
      ].map(([x, z], i) => (
        <mesh key={`side-pillar-${i}`} castShadow position={[x, HEIGHT / 2, z]}>
          <boxGeometry args={[0.08, HEIGHT, 0.08]} />
          <meshStandardMaterial color={COLORS.SILVER_METAL} metalness={0.88} roughness={0.2} />
        </mesh>
      ))}

      {/* === 顶部铝型材框架 === */}
      {/* 主横梁 前后 */}
      {[-HALF_DEPTH + 0.5, HALF_DEPTH - 0.5].map((z, i) => (
        <mesh key={`beam-z-${i}`} castShadow position={[0, HEIGHT - 0.12, z]}>
          <boxGeometry args={[WIDTH - 0.5, 0.12, 0.12]} />
          <meshStandardMaterial color={COLORS.SILVER_METAL} metalness={0.9} roughness={0.15} />
        </mesh>
      ))}
      {/* 主横梁 左右 */}
      {[-HALF_WIDTH + 0.5, HALF_WIDTH - 0.5].map((x, i) => (
        <mesh key={`beam-x-${i}`} castShadow position={[x, HEIGHT - 0.12, 0]}>
          <boxGeometry args={[0.12, 0.12, DEPTH - 0.5]} />
          <meshStandardMaterial color={COLORS.SILVER_METAL} metalness={0.9} roughness={0.15} />
        </mesh>
      ))}
      {/* 次横梁 - 东西向加强 */}
      {[-3, 0, 3].map((z, i) => (
        <mesh key={`sub-beam-${i}`} position={[0, HEIGHT - 0.08, z]}>
          <boxGeometry args={[WIDTH - 1, 0.05, 0.05]} />
          <meshStandardMaterial color={COLORS.SILVER_METAL} metalness={0.8} roughness={0.3} />
        </mesh>
      ))}

      {/* === 顶部软膜天花 - 分块式 === */}
      {/* 中央区 */}
      <mesh position={[0, HEIGHT - 0.35, 0]}>
        <planeGeometry args={[6, 4]} />
        <meshStandardMaterial
          color="#f4f6f4"
          emissive="#e8f0ed"
          emissiveIntensity={1.0}
          side={2}
          transparent
          opacity={0.88}
          roughness={0.6}
        />
      </mesh>
      {/* 四展区天花 */}
      {[
        [-6.5, -4, 4, 2.5],
        [6.5, -4, 4, 2.5],
        [-6.5, 4, 4, 2.5],
        [6.5, 4, 4, 2.5],
        [0, 5.5, 7, 1.8],
      ].map(([x, z, w, d], i) => (
        <mesh key={`ceiling-${i}`} position={[x, HEIGHT - 0.3, z]}>
          <planeGeometry args={[w, d]} />
          <meshStandardMaterial
            color="#f4f6f4"
            emissive="#e8f0ed"
            emissiveIntensity={0.75}
            side={2}
            transparent
            opacity={0.85}
            roughness={0.6}
          />
        </mesh>
      ))}

      {/* === 顶部吊旗/横幅 (增强展会氛围) === */}
      {/* 中央区上方 吊旗 */}
      <mesh position={[-3, HEIGHT - 1.2, -3]}>
        <planeGeometry args={[1.0, 1.4]} />
        <meshStandardMaterial color={COLORS.CELADON_GREEN} side={2} roughness={0.7} />
      </mesh>
      <mesh position={[3, HEIGHT - 1.2, 3]}>
        <planeGeometry args={[1.0, 1.4]} />
        <meshStandardMaterial color={COLORS.LIGHT_BLUE_GREEN} side={2} roughness={0.7} />
      </mesh>

      {/* === 底部边界发光线 (空间轮廓) === */}
      {[
        { pos: [0, 0.015, -HALF_DEPTH], size: [WIDTH, 0.02, 0.02] },
        { pos: [0, 0.015, HALF_DEPTH], size: [WIDTH, 0.02, 0.02] },
        { pos: [-HALF_WIDTH, 0.015, 0], size: [0.02, 0.02, DEPTH] },
        { pos: [HALF_WIDTH, 0.015, 0], size: [0.02, 0.02, DEPTH] },
      ].map((item, i) => (
        <mesh key={`border-${i}`} position={item.pos}>
          <boxGeometry args={item.size} />
          <meshStandardMaterial
            color={COLORS.CELADON_GREEN}
            emissive={COLORS.CELADON_GREEN}
            emissiveIntensity={1.6}
          />
        </mesh>
      ))}
    </group>
  )
}
