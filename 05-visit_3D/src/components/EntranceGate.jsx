import React, { useRef } from 'react'
import { Text, Float } from '@react-three/drei'
import { SPACE, COLORS, ENTRANCE } from '../config/spaceConfig'

/**
 * 入口门头 v2 - 专业临时展会风格
 * 高度 4.2m (高层门头,非全6m)
 * 铝型材立柱 (0.12m方柱) + 软膜灯箱 + LED
 */
export default function EntranceGate() {
  const glowRef = useRef()
  const { HALF_DEPTH } = SPACE

  return (
    <group name="entrance-gate" position={[ENTRANCE.position.x, 0, ENTRANCE.position.z]}>
      {/* === 铝型材立柱 (0.12m方柱, 到4.0m) === */}
      <mesh castShadow position={[-ENTRANCE.width / 2 - 0.2, 2.0, 0]}>
        <boxGeometry args={[0.12, 4.0, 0.12]} />
        <meshStandardMaterial
          color={COLORS.SILVER_METAL}
          metalness={0.9}
          roughness={0.15}
        />
      </mesh>
      <mesh castShadow position={[ENTRANCE.width / 2 + 0.2, 2.0, 0]}>
        <boxGeometry args={[0.12, 4.0, 0.12]} />
        <meshStandardMaterial
          color={COLORS.SILVER_METAL}
          metalness={0.9}
          roughness={0.15}
        />
      </mesh>

      {/* === 铝型材斜撑 (增强结构感) === */}
      <mesh position={[-ENTRANCE.width / 2 - 0.3, 1.0, 0]} rotation={[0, 0, 0.35]}>
        <boxGeometry args={[0.04, 1.5, 0.04]} />
        <meshStandardMaterial color={COLORS.SILVER_METAL} metalness={0.85} roughness={0.2} />
      </mesh>
      <mesh position={[ENTRANCE.width / 2 + 0.3, 1.0, 0]} rotation={[0, 0, -0.35]}>
        <boxGeometry args={[0.04, 1.5, 0.04]} />
        <meshStandardMaterial color={COLORS.SILVER_METAL} metalness={0.85} roughness={0.2} />
      </mesh>

      {/* === 门头主体 - 4.2m, 软膜灯箱 === */}
      <group position={[0, 4.1, 0]}>
        {/* 铝型材横梁 */}
        <mesh castShadow position={[0, 0.2, -0.04]}>
          <boxGeometry args={[ENTRANCE.width + 1.0, 0.16, 0.16]} />
          <meshStandardMaterial color={COLORS.SILVER_METAL} metalness={0.9} roughness={0.15} />
        </mesh>

        {/* 主灯箱 - 软膜效果 */}
        <mesh position={[0, 0, 0.02]}>
          <boxGeometry args={[ENTRANCE.width + 0.6, 1.4, 0.06]} />
          <meshStandardMaterial
            color="#ffffff"
            transparent
            opacity={0.88}
            roughness={0.4}
            metalness={0}
            emissive={COLORS.CELADON_GREEN}
            emissiveIntensity={0.15}
          />
        </mesh>

        {/* 门头发光边框 */}
        <mesh ref={glowRef} position={[0, 0.72, 0.06]}>
          <boxGeometry args={[ENTRANCE.width + 0.5, 0.04, 0.03]} />
          <meshStandardMaterial
            color={COLORS.CELADON_GREEN}
            emissive={COLORS.CELADON_GREEN}
            emissiveIntensity={1.5}
          />
        </mesh>
        <mesh position={[0, -0.72, 0.06]}>
          <boxGeometry args={[ENTRANCE.width + 0.5, 0.04, 0.03]} />
          <meshStandardMaterial
            color={COLORS.CELADON_GREEN}
            emissive={COLORS.CELADON_GREEN}
            emissiveIntensity={1.5}
          />
        </mesh>
        <mesh position={[-ENTRANCE.width / 2 - 0.22, 0, 0.06]}>
          <boxGeometry args={[0.04, 1.44, 0.03]} />
          <meshStandardMaterial color={COLORS.CELADON_GREEN} emissive={COLORS.CELADON_GREEN} emissiveIntensity={1.2} />
        </mesh>
        <mesh position={[ENTRANCE.width / 2 + 0.22, 0, 0.06]}>
          <boxGeometry args={[0.04, 1.44, 0.03]} />
          <meshStandardMaterial color={COLORS.CELADON_GREEN} emissive={COLORS.CELADON_GREEN} emissiveIntensity={1.2} />
        </mesh>

        {/* === 文字内容 === */}
        <Float speed={1.2} rotationIntensity={0} floatIntensity={0.15}>
          <Text
            position={[0, 0.25, 0.08]}
            fontSize={0.52}
            color="#1a3a35"
            anchorX="center"
            anchorY="middle"
            letterSpacing={0.25}
          >
            浙江馆
            <meshStandardMaterial
              color="#1a3a35"
              emissive={COLORS.CELADON_GREEN}
              emissiveIntensity={0.12}
            />
          </Text>
        </Float>

        <Text
          position={[0, -0.15, 0.08]}
          fontSize={0.18}
          color={COLORS.CELADON_GREEN}
          anchorX="center"
          anchorY="middle"
          letterSpacing={0.35}
        >
          ZHEJIANG
          <meshStandardMaterial
            color={COLORS.CELADON_GREEN}
            emissive={COLORS.CELADON_GREEN}
            emissiveIntensity={0.6}
          />
        </Text>

        <mesh position={[0, -0.38, 0.08]}>
          <planeGeometry args={[2.4, 0.015]} />
          <meshStandardMaterial
            color={COLORS.CELADON_GREEN}
            emissive={COLORS.CELADON_GREEN}
            emissiveIntensity={0.8}
            transparent
            opacity={0.8}
          />
        </mesh>

        <Float speed={1.5} rotationIntensity={0} floatIntensity={0.12}>
          <Text
            position={[0, -0.55, 0.08]}
            fontSize={0.22}
            color="#3a6a60"
            anchorX="center"
            anchorY="middle"
            letterSpacing={0.2}
          >
            浙韵焕新
            <meshStandardMaterial
              color="#3a6a60"
              emissive={COLORS.LIGHT_BLUE_GREEN}
              emissiveIntensity={0.18}
            />
          </Text>
        </Float>
      </group>

      {/* === 入口地面引导 === */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.012, 0.8]}>
        <planeGeometry args={[ENTRANCE.width, 2.5]} />
        <meshStandardMaterial
          color={COLORS.CELADON_GREEN}
          emissive={COLORS.CELADON_GREEN}
          emissiveIntensity={0.2}
          transparent
          opacity={0.25}
        />
      </mesh>

      {/* 入口欢迎光带 */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.025, 0]}>
        <planeGeometry args={[ENTRANCE.width + 0.8, 0.06]} />
        <meshStandardMaterial
          color={COLORS.CELADON_GREEN}
          emissive={COLORS.CELADON_GREEN}
          emissiveIntensity={2}
          transparent
          opacity={0.9}
        />
      </mesh>
    </group>
  )
}
