import React, { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Text } from '@react-three/drei'
import * as THREE from 'three'
import { ZONES, COLORS, HERITAGE_INDUSTRIES } from '../config/spaceConfig'

/* ========================================================================
   展区基础组件 v2 - 参数化三层高度系统
   低层 0-1.2m: 展台/产品
   中层 2.4-3.5m: 主展墙/灯箱
   高层 4-5.5m: 门头/吊楣
   材质: 铝型材/木结构/软膜/灯箱/LED/玻璃/金属网
======================================================================== */
function ZoneBase({ zone, children, showPodium = true }) {
  const { position, size, color, name, subtitle } = zone
  const wallHeight = 2.8    // 中层主展墙
  const headerHeight = 4.2  // 高层门头吊楣
  const podiumHeight = 0.9  // 低层展台

  return (
    <group position={[position.x, 0, position.z]}>
      {/* === 展区地面标识 === */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.008, 0]}>
        <planeGeometry args={[size.width, size.depth]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.05}
          transparent
          opacity={0.12}
          roughness={0.8}
        />
      </mesh>

      {/* === 中层 主展墙 (2.8m) === */}
      {/* 主背板 - 白木板基层 */}
      <mesh castShadow receiveShadow position={[0, wallHeight / 2, -size.depth / 2 + 0.04]}>
        <boxGeometry args={[size.width, wallHeight, 0.06]} />
        <meshStandardMaterial
          color="#f2ede4"
          roughness={0.55}
          metalness={0.02}
        />
      </mesh>
      {/* 背板前 灯箱层 - 半透软膜效果 */}
      <mesh position={[0, wallHeight / 2, -size.depth / 2 + 0.075]}>
        <boxGeometry args={[size.width - 0.2, wallHeight - 0.2, 0.02]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.22}
          roughness={0.4}
          metalness={0}
          emissive={color}
          emissiveIntensity={0.05}
        />
      </mesh>
      {/* 背板底部 LED 洗墙光带 */}
      <mesh position={[0, 0.08, -size.depth / 2 + 0.06]}>
        <boxGeometry args={[size.width - 0.3, 0.04, 0.02]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={1.5}
          transparent
          opacity={0.9}
        />
      </mesh>
      {/* 背板顶部 LED 光带 */}
      <mesh position={[0, wallHeight - 0.08, -size.depth / 2 + 0.06]}>
        <boxGeometry args={[size.width - 0.3, 0.04, 0.02]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={1.2}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* === 侧展墙 (朝向中央通道,形成围合感) === */}
      {/* 左侧展墙 */}
      <mesh castShadow receiveShadow position={[-size.width / 2 + 0.03, wallHeight / 2, 0]}>
        <boxGeometry args={[0.05, wallHeight, size.depth]} />
        <meshStandardMaterial color="#f2ede4" roughness={0.55} metalness={0.02} />
      </mesh>
      <mesh position={[-size.width / 2 + 0.06, wallHeight - 0.08, 0]}>
        <boxGeometry args={[0.02, 0.04, size.depth - 0.3]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.1} transparent opacity={0.85} />
      </mesh>
      {/* 右侧展墙 */}
      <mesh castShadow receiveShadow position={[size.width / 2 - 0.03, wallHeight / 2, 0]}>
        <boxGeometry args={[0.05, wallHeight, size.depth]} />
        <meshStandardMaterial color="#f2ede4" roughness={0.55} metalness={0.02} />
      </mesh>
      <mesh position={[size.width / 2 - 0.06, wallHeight - 0.08, 0]}>
        <boxGeometry args={[0.02, 0.04, size.depth - 0.3]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.1} transparent opacity={0.85} />
      </mesh>

      {/* === 侧墙图文展板 (朝向中央通道,双面可见,增强视觉层次) === */}
      {/* 左侧 */}
      <group position={[-size.width / 2 - 0.03, 1.5, 0]} rotation={[0, Math.PI / 2, 0]}>
        {/* 主图板 */}
        <mesh>
          <planeGeometry args={[size.depth - 0.5, 1.6]} />
          <meshStandardMaterial color="#f5f1ea" roughness={0.45} metalness={0.02} side={THREE.DoubleSide} />
        </mesh>
        {/* 顶部标题栏 */}
        <mesh position={[0, 0.55, 0.01]}>
          <planeGeometry args={[size.depth - 0.7, 0.14]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.7} transparent opacity={0.85} side={THREE.DoubleSide} />
        </mesh>
        {/* 中部信息区 */}
        <mesh position={[0, 0, 0.01]}>
          <planeGeometry args={[size.depth - 0.7, 0.7]} />
          <meshStandardMaterial color="#ffffff" roughness={0.5} side={THREE.DoubleSide} />
        </mesh>
        {/* 底部数据条 */}
        <mesh position={[0, -0.65, 0.01]}>
          <planeGeometry args={[size.depth - 0.7, 0.12]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.9} transparent opacity={0.75} side={THREE.DoubleSide} />
        </mesh>
        {/* 分割线 */}
        <mesh position={[0, 0.35, 0.012]}>
          <planeGeometry args={[size.depth - 0.7, 0.01]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} transparent opacity={0.9} side={THREE.DoubleSide} />
        </mesh>
      </group>
      {/* 右侧 */}
      <group position={[size.width / 2 + 0.03, 1.5, 0]} rotation={[0, -Math.PI / 2, 0]}>
        <mesh>
          <planeGeometry args={[size.depth - 0.5, 1.6]} />
          <meshStandardMaterial color="#f5f1ea" roughness={0.45} metalness={0.02} side={THREE.DoubleSide} />
        </mesh>
        <mesh position={[0, 0.55, 0.01]}>
          <planeGeometry args={[size.depth - 0.7, 0.14]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.7} transparent opacity={0.85} side={THREE.DoubleSide} />
        </mesh>
        <mesh position={[0, 0, 0.01]}>
          <planeGeometry args={[size.depth - 0.7, 0.7]} />
          <meshStandardMaterial color="#ffffff" roughness={0.5} side={THREE.DoubleSide} />
        </mesh>
        <mesh position={[0, -0.65, 0.01]}>
          <planeGeometry args={[size.depth - 0.7, 0.12]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.9} transparent opacity={0.75} side={THREE.DoubleSide} />
        </mesh>
        <mesh position={[0, 0.35, 0.012]}>
          <planeGeometry args={[size.depth - 0.7, 0.01]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.8} transparent opacity={0.9} side={THREE.DoubleSide} />
        </mesh>
      </group>

      {/* === 高层 门头吊楣 (4.2m) === */}
      {/* 铝型材横梁 */}
      <mesh castShadow position={[0, headerHeight, -size.depth / 2 + 0.08]}>
        <boxGeometry args={[size.width + 0.4, 0.18, 0.18]} />
        <meshStandardMaterial
          color={COLORS.SILVER_METAL}
          metalness={0.9}
          roughness={0.15}
        />
      </mesh>
      {/* 软膜灯箱条幅 */}
      <mesh position={[0, headerHeight - 0.14, -size.depth / 2 + 0.12]}>
        <boxGeometry args={[size.width, 0.22, 0.04]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.8}
          transparent
          opacity={0.85}
          side={THREE.DoubleSide}
        />
      </mesh>
      {/* 吊楣延伸到两侧 - 形成门型 */}
      <mesh position={[-size.width / 2 + 0.02, headerHeight, 0]}>
        <boxGeometry args={[0.05, 0.18, size.depth]} />
        <meshStandardMaterial color={COLORS.SILVER_METAL} metalness={0.9} roughness={0.15} />
      </mesh>
      <mesh position={[size.width / 2 - 0.02, headerHeight, 0]}>
        <boxGeometry args={[0.05, 0.18, size.depth]} />
        <meshStandardMaterial color={COLORS.SILVER_METAL} metalness={0.9} roughness={0.15} />
      </mesh>
      {/* 侧吊楣软膜 */}
      <mesh position={[-size.width / 2 + 0.04, headerHeight - 0.14, 0]} rotation={[0, Math.PI / 2, 0]}>
        <boxGeometry args={[size.depth - 0.2, 0.22, 0.03]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} transparent opacity={0.8} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[size.width / 2 - 0.04, headerHeight - 0.14, 0]} rotation={[0, Math.PI / 2, 0]}>
        <boxGeometry args={[size.depth - 0.2, 0.22, 0.03]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} transparent opacity={0.8} side={THREE.DoubleSide} />
      </mesh>

      {/* === 展区标题 (中层顶部) === */}
      <group position={[0, wallHeight - 0.35, -size.depth / 2 + 0.09]}>
        <Text
          position={[0, 0.12, 0]}
          fontSize={0.32}
          color="#1a2a30"
          anchorX="center"
          anchorY="middle"
          letterSpacing={0.15}
          maxWidth={size.width - 1}
        >
          {name}
          <meshStandardMaterial color="#1a2a30" />
        </Text>
        <mesh position={[0, -0.08, 0]}>
          <planeGeometry args={[Math.min(size.width * 0.4, 3.2), 0.012]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={1.0}
            transparent
            opacity={0.8}
          />
        </mesh>
        <Text
          position={[0, -0.24, 0]}
          fontSize={0.12}
          color={color}
          anchorX="center"
          anchorY="middle"
          letterSpacing={0.1}
          maxWidth={size.width - 1}
        >
          {subtitle}
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.3}
          />
        </Text>
      </group>

      {/* === 低层 展柜系统 (0.9m 木质/金属基座 + 1.0m 玻璃展柜) === */}
      {showPodium && (
        <group position={[0, 0, size.depth / 2 - 0.85]}>
          {/* 基座 - 实木基层 */}
          <mesh castShadow receiveShadow position={[0, podiumHeight / 2, 0]}>
            <boxGeometry args={[size.width * 0.55, podiumHeight, 0.7]} />
            <meshStandardMaterial
              color="#8c6b4a"
              roughness={0.7}
              metalness={0.05}
            />
          </mesh>
          {/* 基座顶面 - 铝塑板面层 */}
          <mesh receiveShadow position={[0, podiumHeight + 0.006, 0]}>
            <boxGeometry args={[size.width * 0.56, 0.012, 0.72]} />
            <meshStandardMaterial
              color="#f5f3ee"
              roughness={0.35}
              metalness={0.08}
            />
          </mesh>
          {/* 基座底部 LED 洗地发光条 */}
          <mesh position={[0, 0.035, 0.36]}>
            <boxGeometry args={[size.width * 0.5, 0.02, 0.012]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={1.4}
              transparent
              opacity={0.85}
            />
          </mesh>

          {/* 玻璃展柜 - 位于基座之上 (高 1.0m) */}
          <group position={[0, podiumHeight, 0]}>
            {/* 金属框架 - 四角立柱 */}
            {[
              [-size.width * 0.26, -0.3],
              [size.width * 0.26, -0.3],
              [-size.width * 0.26, 0.3],
              [size.width * 0.26, 0.3],
            ].map(([x, z], i) => (
              <mesh key={`case-post-${i}`} castShadow position={[x, 0.5, z]}>
                <boxGeometry args={[0.04, 1.0, 0.04]} />
                <meshStandardMaterial
                  color={COLORS.SILVER_METAL}
                  metalness={0.9}
                  roughness={0.2}
                />
              </mesh>
            ))}
            {/* 展柜顶梁 */}
            <mesh position={[0, 1.0, 0]}>
              <boxGeometry args={[size.width * 0.54, 0.04, 0.64]} />
              <meshStandardMaterial
                color={COLORS.SILVER_METAL}
                metalness={0.88}
                roughness={0.2}
              />
            </mesh>
            {/* 玻璃正面 */}
            <mesh position={[0, 0.5, -0.31]}>
              <boxGeometry args={[size.width * 0.52, 0.96, 0.01]} />
              <meshPhysicalMaterial
                color="#eaf3f5"
                transparent
                opacity={0.15}
                roughness={0.05}
                metalness={0}
                transmission={0.85}
                thickness={0.02}
                ior={1.45}
                side={THREE.DoubleSide}
              />
            </mesh>
            {/* 玻璃背面 */}
            <mesh position={[0, 0.5, 0.31]}>
              <boxGeometry args={[size.width * 0.52, 0.96, 0.01]} />
              <meshPhysicalMaterial
                color="#eaf3f5"
                transparent
                opacity={0.15}
                roughness={0.05}
                metalness={0}
                transmission={0.85}
                thickness={0.02}
                ior={1.45}
                side={THREE.DoubleSide}
              />
            </mesh>
            {/* 展柜内托台 */}
            <mesh position={[0, 0.08, 0]}>
              <boxGeometry args={[size.width * 0.48, 0.025, 0.56]} />
              <meshStandardMaterial
                color="#fafaf5"
                roughness={0.3}
                metalness={0.1}
                emissive={color}
                emissiveIntensity={0.05}
              />
            </mesh>
            {/* 展柜内顶部 LED 照明 */}
            <mesh position={[0, 0.97, 0]}>
              <boxGeometry args={[size.width * 0.48, 0.015, 0.56]} />
              <meshStandardMaterial
                color="#fffbe8"
                emissive="#fff5d0"
                emissiveIntensity={1.2}
                transparent
                opacity={0.95}
              />
            </mesh>
            {/* 展柜内 3个产品基座 */}
            {[
              { x: -size.width * 0.15, label: '青瓷', shape: 'cyl' },
              { x: 0, label: '丝绸', shape: 'torus' },
              { x: size.width * 0.15, label: '茶叶', shape: 'box' },
            ].map((p, i) => (
              <group key={`product-${i}`} position={[p.x, 0.18, 0]}>
                <mesh castShadow>
                  <cylinderGeometry args={[0.1, 0.12, 0.06, 16]} />
                  <meshStandardMaterial color={color} roughness={0.3} metalness={0.4} />
                </mesh>
                {p.shape === 'cyl' && (
                  <mesh position={[0, 0.18, 0]} castShadow>
                    <cylinderGeometry args={[0.08, 0.06, 0.3, 24]} />
                    <meshStandardMaterial color={COLORS.CELADON_GREEN} roughness={0.25} metalness={0.15} emissive={COLORS.CELADON_GREEN} emissiveIntensity={0.05} />
                  </mesh>
                )}
                {p.shape === 'torus' && (
                  <mesh position={[0, 0.18, 0]} rotation={[Math.PI / 2, 0, 0]} castShadow>
                    <torusGeometry args={[0.11, 0.025, 12, 32]} />
                    <meshStandardMaterial color="#d8b582" roughness={0.35} metalness={0.3} />
                  </mesh>
                )}
                {p.shape === 'box' && (
                  <mesh position={[0, 0.18, 0]} castShadow>
                    <boxGeometry args={[0.18, 0.28, 0.12]} />
                    <meshStandardMaterial color="#6a4a2a" roughness={0.7} metalness={0.05} />
                  </mesh>
                )}
              </group>
            ))}
          </group>
        </group>
      )}

      {/* 子内容 */}
      {children}
    </group>
  )
}

/* ========================================================================
   01 源·文脉赓续 - 历史经典产业数字地图墙
======================================================================== */
function OriginZone() {
  const zone = ZONES.ORIGIN
  const itemsRef = useRef([])

  useFrame((state) => {
    const t = state.clock.elapsedTime
    itemsRef.current.forEach((item, i) => {
      if (item) {
        item.scale.setScalar(1 + Math.sin(t * 1.2 + i * 0.3) * 0.04)
      }
    })
  })

  const layout = []
  HERITAGE_INDUSTRIES.forEach((item, i) => {
    const col = i % 4
    const row = Math.floor(i / 4)
    layout.push({
      ...item,
      x: -2.0 + col * 1.35,
      y: 1.5 - row * 0.85,
    })
  })

  return (
    <ZoneBase zone={zone}>
      <group position={[0, 0, -zone.size.depth / 2 + 0.1]}>
        {/* 地图背景层 */}
        <mesh position={[0, 1.5, 0]}>
          <planeGeometry args={[zone.size.width - 0.6, 2.6]} />
          <meshStandardMaterial
            color="#e8f0ed"
            transparent
            opacity={0.9}
            roughness={0.6}
          />
        </mesh>

        {/* 产业卡片网格 */}
        {layout.map((item, i) => (
          <group
            key={`heritage-${i}`}
            position={[item.x, item.y, 0.02]}
            ref={(el) => (itemsRef.current[i] = el)}
          >
            {/* 卡片底板 */}
            <mesh position={[0, 0, 0]}>
              <boxGeometry args={[1.2, 0.72, 0.035]} />
              <meshStandardMaterial
                color="#ffffff"
                roughness={0.35}
                metalness={0.05}
              />
            </mesh>
            {/* 卡片发光边框 */}
            <mesh position={[0, 0, 0.022]}>
              <planeGeometry args={[1.14, 0.66]} />
              <meshBasicMaterial
                color={zone.color}
                transparent
                opacity={0.12}
              />
            </mesh>
            <mesh position={[0, -0.35, 0.028]}>
              <planeGeometry args={[1.18, 0.012]} />
              <meshStandardMaterial
                color={zone.color}
                emissive={zone.color}
                emissiveIntensity={0.8}
                transparent
                opacity={0.7}
              />
            </mesh>

            {/* 简化产业图标 */}
            <group position={[-0.32, 0.08, 0.035]}>
              {i % 5 === 0 && (
                <mesh>
                  <icosahedronGeometry args={[0.16, 0]} />
                  <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.2} roughness={0.4} />
                </mesh>
              )}
              {i % 5 === 1 && (
                <mesh>
                  <torusGeometry args={[0.14, 0.04, 12, 24]} />
                  <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.2} roughness={0.4} />
                </mesh>
              )}
              {i % 5 === 2 && (
                <mesh>
                  <cylinderGeometry args={[0.11, 0.09, 0.28, 16]} />
                  <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.2} roughness={0.4} />
                </mesh>
              )}
              {i % 5 === 3 && (
                <mesh rotation={[0, 0, Math.PI / 4]}>
                  <boxGeometry args={[0.22, 0.22, 0.06]} />
                  <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.2} roughness={0.4} />
                </mesh>
              )}
              {i % 5 === 4 && (
                <mesh>
                  <octahedronGeometry args={[0.18, 0]} />
                  <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.2} roughness={0.4} />
                </mesh>
              )}
            </group>

            <Text position={[0.25, 0.1, 0.035]} fontSize={0.14} color="#1a3030" anchorX="left" anchorY="middle">
              {item.name}
              <meshStandardMaterial color="#1a3030" />
            </Text>
            <Text position={[0.25, -0.1, 0.035]} fontSize={0.08} color={zone.color} anchorX="left" anchorY="middle">
              {item.region}
              <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.2} />
            </Text>
          </group>
        ))}
      </group>
    </ZoneBase>
  )
}

/* ========================================================================
   02 涌·科技筑能 - 数字信息流和互动LED屏幕
======================================================================== */
function SurgeZone() {
  const zone = ZONES.SURGE
  const dataLinesRef = useRef([])
  const screenRef = useRef()

  useFrame((state) => {
    const t = state.clock.elapsedTime
    dataLinesRef.current.forEach((line, i) => {
      if (line) {
        const offset = (t * 0.3 + i * 0.1) % 1
        line.material.uniforms.time.value = t
        line.material.uniforms.offset.value = offset
      }
    })
    if (screenRef.current) {
      screenRef.current.material.uniforms.time.value = t
    }
  })

  const dataLines = Array.from({ length: 14 }, (_, i) => ({
    x: -2.2 + i * 0.32,
    height: 0.6 + Math.random() * 1.8,
  }))

  const dataFlowShader = {
    uniforms: {
      time: { value: 0 },
      offset: { value: 0 },
      color: { value: new THREE.Color(zone.color) },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float time;
      uniform float offset;
      uniform vec3 color;
      varying vec2 vUv;
      void main() {
        float flow = fract(vUv.y * 5.0 - time * 0.5 + offset);
        float line = smoothstep(0.0, 0.05, flow) * smoothstep(0.12, 0.08, flow);
        float glow = smoothstep(0.0, 0.3, vUv.y) * smoothstep(1.0, 0.7, vUv.y);
        gl_FragColor = vec4(color * (0.6 + line * 1.5), glow * 0.85);
      }
    `,
    transparent: true,
  }

  const screenShader = {
    uniforms: {
      time: { value: 0 },
      color: { value: new THREE.Color(zone.color) },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float time;
      uniform vec3 color;
      varying vec2 vUv;
      void main() {
        float gridX = smoothstep(0.0, 0.02, fract(vUv.x * 40.0)) * smoothstep(1.0, 0.98, fract(vUv.x * 40.0));
        float gridY = smoothstep(0.0, 0.02, fract(vUv.y * 25.0)) * smoothstep(1.0, 0.98, fract(vUv.y * 25.0));
        float grid = gridX * gridY;
        float wave1 = sin(vUv.x * 20.0 + time * 1.5) * 0.5 + 0.5;
        float wave2 = sin(vUv.x * 15.0 + time * 2.1 + 1.0) * 0.5 + 0.5;
        float waves = step(vUv.y, wave1 * 0.6) * 0.3 + step(vUv.y, wave2 * 0.4) * 0.2;
        vec3 bg1 = vec3(0.02, 0.06, 0.1);
        vec3 bg2 = vec3(0.05, 0.12, 0.18);
        vec3 bg = mix(bg1, bg2, vUv.y);
        vec3 finalColor = bg + color * (waves + grid * 0.15);
        gl_FragColor = vec4(finalColor, 1.0);
      }
    `,
  }

  return (
    <ZoneBase zone={zone}>
      <group position={[0, 0, -zone.size.depth / 2 + 0.1]}>
        {/* LED大屏幕 */}
        <mesh ref={screenRef} position={[0, 1.7, 0]}>
          <planeGeometry args={[zone.size.width - 0.8, 1.8]} />
          <shaderMaterial attach="material" args={[screenShader]} />
        </mesh>
        {/* 屏幕边框 - 铝型材 */}
        <mesh position={[0, 1.7, -0.01]}>
          <boxGeometry args={[zone.size.width - 0.5, 2.0, 0.04]} />
          <meshStandardMaterial
            color={COLORS.SILVER_METAL}
            metalness={0.9}
            roughness={0.2}
          />
        </mesh>

        {/* 数字信息流 */}
        <group position={[0, 0.5, 0.02]}>
          {dataLines.map((line, i) => (
            <mesh
              key={`dataline-${i}`}
              ref={(el) => (dataLinesRef.current[i] = el)}
              position={[line.x, line.height / 2, 0]}
            >
              <planeGeometry args={[0.025, line.height]} />
              <shaderMaterial attach="material" args={[dataFlowShader]} />
            </mesh>
          ))}
        </group>

        {/* 科技指标数据块 */}
        <group position={[0, 0.2, 0.02]}>
          {[
            { label: '智能制造', value: 'AI+', x: -1.8 },
            { label: '数字转型', value: '100%', x: 0 },
            { label: '技术突破', value: '5G+', x: 1.8 },
          ].map((item, i) => (
            <group key={`stat-${i}`} position={[item.x, 0, 0]}>
              <mesh>
                <boxGeometry args={[1.4, 0.4, 0.04]} />
                <meshStandardMaterial color="#ffffff" roughness={0.3} />
              </mesh>
              <Text position={[0, 0.06, 0.03]} fontSize={0.18} color={zone.color} anchorX="center">
                {item.value}
                <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.5} />
              </Text>
              <Text position={[0, -0.1, 0.03]} fontSize={0.08} color="#445566" anchorX="center">
                {item.label}
                <meshStandardMaterial color="#445566" />
              </Text>
            </group>
          ))}
        </group>
      </group>
    </ZoneBase>
  )
}

/* ========================================================================
   03 涟·聚链共生 - 连续发光曲线表现产业链
======================================================================== */
function RippleZone() {
  const zone = ZONES.RIPPLE
  const chainCurveRef = useRef()

  useFrame((state) => {
    const t = state.clock.elapsedTime
    if (chainCurveRef.current) {
      chainCurveRef.current.material.uniforms.time.value = t
    }
  })

  const chainNodes = [
    { name: '原材料', sub: 'Raw Materials', x: -2.2, y: 1.6 },
    { name: '生产制造', sub: 'Production', x: -0.8, y: 2.0 },
    { name: '销售渠道', sub: 'Sales', x: 0.8, y: 1.6 },
    { name: '服务运营', sub: 'Service', x: 2.2, y: 2.0 },
    { name: '品牌升级', sub: 'Branding', x: -1.4, y: 0.9 },
    { name: '产业集群', sub: 'Clusters', x: 0, y: 1.1 },
    { name: '全球贸易', sub: 'Global Trade', x: 1.4, y: 0.9 },
  ]

  const chainShader = {
    uniforms: {
      time: { value: 0 },
      color: { value: new THREE.Color(zone.color) },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform float time;
      uniform vec3 color;
      varying vec2 vUv;
      void main() {
        float flow = fract(vUv.x * 3.0 - time * 0.4);
        float pulse = smoothstep(0.0, 0.1, flow) * smoothstep(0.3, 0.15, flow);
        float base = 0.35 + pulse * 0.65;
        gl_FragColor = vec4(color, base);
      }
    `,
    transparent: true,
  }

  return (
    <ZoneBase zone={zone}>
      <group position={[0, 0, -zone.size.depth / 2 + 0.1]}>
        {/* 背景装饰 - 涟漪同心圆 */}
        {[0.35, 0.7, 1.05, 1.4].map((r, i) => (
          <mesh key={`ripple-bg-${i}`} position={[0, 1.8, -0.01 + i * 0.005]}>
            <ringGeometry args={[r * 1.1, r * 1.1 + 0.012, 64]} />
            <meshBasicMaterial color={zone.color} transparent opacity={0.1 - i * 0.015} side={THREE.DoubleSide} />
          </mesh>
        ))}

        {/* 产业链发光连线 */}
        <mesh ref={chainCurveRef} position={[0, 2.1, 0.01]}>
          <planeGeometry args={[zone.size.width - 0.8, 0.03]} />
          <shaderMaterial attach="material" args={[chainShader]} />
        </mesh>
        <mesh position={[0, 1.2, 0.01]}>
          <planeGeometry args={[zone.size.width - 1.2, 0.025]} />
          <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.8} transparent opacity={0.7} />
        </mesh>
        <mesh position={[-1.4, 1.65, 0.01]}>
          <planeGeometry args={[0.025, 0.8]} />
          <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.8} transparent opacity={0.7} />
        </mesh>
        <mesh position={[0, 1.85, 0.01]}>
          <planeGeometry args={[0.025, 1.0]} />
          <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.8} transparent opacity={0.7} />
        </mesh>
        <mesh position={[1.4, 1.65, 0.01]}>
          <planeGeometry args={[0.025, 0.8]} />
          <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.8} transparent opacity={0.7} />
        </mesh>

        {/* 产业链节点 */}
        {chainNodes.map((node, i) => (
          <group key={`chain-node-${i}`} position={[node.x, node.y, 0.03]}>
            <mesh>
              <ringGeometry args={[0.24, 0.26, 32]} />
              <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={1} transparent opacity={0.8} />
            </mesh>
            <mesh>
              <circleGeometry args={[0.2, 32]} />
              <meshStandardMaterial color="#ffffff" emissive={zone.color} emissiveIntensity={0.15} roughness={0.3} />
            </mesh>
            <mesh position={[0, 0, 0.01]}>
              <circleGeometry args={[0.06, 20]} />
              <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={1.8} />
            </mesh>
            <Text position={[0, -0.42, 0]} fontSize={0.11} color="#1a3030" anchorX="center">
              {node.name}
              <meshStandardMaterial color="#1a3030" />
            </Text>
            <Text position={[0, -0.56, 0]} fontSize={0.065} color={zone.color} anchorX="center">
              {node.sub}
              <meshStandardMaterial color={zone.color} />
            </Text>
          </group>
        ))}
      </group>
    </ZoneBase>
  )
}

/* ========================================================================
   04 汇·生态赋质 - 数据云 / 数字矩阵
======================================================================== */
function ConvergeZone() {
  const zone = ZONES.CONVERGE
  const cubesRef = useRef([])

  useFrame((state) => {
    const t = state.clock.elapsedTime
    cubesRef.current.forEach((cube, i) => {
      if (cube) {
        cube.rotation.x = t * 0.2 + i * 0.1
        cube.rotation.y = t * 0.15 + i * 0.15
        cube.position.y = cube.userData.baseY + Math.sin(t * 0.8 + i * 0.4) * 0.06
      }
    })
  })

  const ecoElements = [
    { name: '政策支持', value: 'Policy', count: 36 },
    { name: '创新平台', value: 'Platforms', count: 48 },
    { name: '人才汇聚', value: 'Talents', count: 64 },
    { name: '高校科研', value: 'Research', count: 52 },
    { name: 'AI赋能', value: 'AI+', count: 28 },
    { name: '数字经济', value: 'Digital', count: 56 },
    { name: '现代服务', value: 'Service', count: 42 },
  ]

  const cubes = Array.from({ length: 30 }, (_, i) => ({
    x: -2.2 + (i % 8) * 0.6,
    baseY: 1.0 + Math.floor(i / 8) * 0.35,
    z: 0.02,
    scale: 0.06 + Math.random() * 0.08,
  }))

  return (
    <ZoneBase zone={zone}>
      <group position={[0, 0, -zone.size.depth / 2 + 0.1]}>
        {/* 背景 - 数字矩阵网格 */}
        <mesh position={[0, 1.5, -0.01]}>
          <planeGeometry args={[zone.size.width - 0.6, 2.6]} />
          <meshStandardMaterial color="#f0f5f8" transparent opacity={0.85} roughness={0.6} />
        </mesh>
        {Array.from({ length: 8 }).map((_, i) => (
          <mesh key={`grid-h-${i}`} position={[0, 0.2 + i * 0.3, 0]}>
            <planeGeometry args={[zone.size.width - 1, 0.006]} />
            <meshBasicMaterial color={zone.color} transparent opacity={0.12} />
          </mesh>
        ))}
        {Array.from({ length: 10 }).map((_, i) => (
          <mesh key={`grid-v-${i}`} position={[-2.4 + i * 0.5, 1.5, 0]}>
            <planeGeometry args={[0.006, 2.5]} />
            <meshBasicMaterial color={zone.color} transparent opacity={0.12} />
          </mesh>
        ))}

        {/* 数字云悬浮方块 */}
        {cubes.map((cube, i) => (
          <mesh
            key={`cloud-cube-${i}`}
            ref={(el) => {
              cubesRef.current[i] = el
              if (el) el.userData.baseY = cube.baseY
            }}
            position={[cube.x, cube.baseY, cube.z]}
            scale={cube.scale}
          >
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial
              color={i % 2 === 0 ? zone.color : COLORS.LIGHT_BLUE_GREEN}
              emissive={i % 2 === 0 ? zone.color : COLORS.LIGHT_BLUE_GREEN}
              emissiveIntensity={0.3}
              transparent
              opacity={0.7}
              metalness={0.4}
              roughness={0.3}
            />
          </mesh>
        ))}

        {/* 生态要素标签 */}
        <group position={[0, 0.35, 0.03]}>
          {ecoElements.map((item, i) => {
            const col = i % 4
            const row = Math.floor(i / 4)
            return (
              <group key={`eco-${i}`} position={[-2.0 + col * 1.35, row === 0 ? 0 : -0.45, 0]}>
                <mesh>
                  <boxGeometry args={[1.15, 0.35, 0.03]} />
                  <meshStandardMaterial color="#ffffff" roughness={0.35} />
                </mesh>
                <mesh position={[-0.52, 0, 0.02]}>
                  <planeGeometry args={[0.03, 0.3]} />
                  <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.8} />
                </mesh>
                <Text position={[-0.3, 0.04, 0.02]} fontSize={0.11} color={zone.color} anchorX="left">
                  {item.count}
                  <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.4} />
                </Text>
                <Text position={[0, 0.04, 0.02]} fontSize={0.09} color="#1a3030" anchorX="center">
                  {item.name}
                  <meshStandardMaterial color="#1a3030" />
                </Text>
              </group>
            )
          })}
        </group>
      </group>
    </ZoneBase>
  )
}

/* ========================================================================
   05 润·境生万象 - 浙韵书房/现代生活场景
======================================================================== */
function NourishZone() {
  const zone = ZONES.NOURISH
  const sceneLightRef = useRef()

  useFrame((state) => {
    const t = state.clock.elapsedTime
    if (sceneLightRef.current) {
      sceneLightRef.current.intensity = 1 + Math.sin(t * 0.6) * 0.15
    }
  })

  return (
    <ZoneBase zone={zone}>
      <group position={[0, 0, -zone.size.depth / 2 + 0.1]}>
        {/* 生活场景背景 - 暖色 */}
        <mesh position={[0, 1.5, -0.01]}>
          <planeGeometry args={[zone.size.width - 0.4, 3.0]} />
          <meshStandardMaterial color="#f8f4ee" roughness={0.7} />
        </mesh>

        {/* 三个场景区域 */}
        {[
          { x: -3.0, title: '浙韵书房', sub: 'Scholar Studio' },
          { x: 0, title: '经典出海', sub: 'Global Reach' },
          { x: 3.0, title: '文化体验', sub: 'Living Culture' },
        ].map((scene, si) => (
          <group key={`scene-${si}`} position={[scene.x, 1.6, 0.03]}>
            <mesh position={[0, 0, -0.01]}>
              <boxGeometry args={[2.6, 2.8, 0.04]} />
              <meshStandardMaterial color="#ffffff" roughness={0.4} />
            </mesh>
            <mesh position={[0, 0, 0]}>
              <planeGeometry args={[2.4, 2.6]} />
              <meshStandardMaterial color={si === 0 ? '#f5efe5' : si === 1 ? '#eef3f7' : '#f5f0f0'} roughness={0.6} />
            </mesh>

            {/* 场景1: 浙韵书房 */}
            {si === 0 && (
              <group position={[0, -0.3, 0.02]}>
                <mesh position={[0, -0.55, 0]}>
                  <boxGeometry args={[2.0, 0.07, 0.35]} />
                  <meshStandardMaterial color="#c4a57b" roughness={0.7} metalness={0.1} />
                </mesh>
                {[-1, 1].map((sx) => (
                  <mesh key={`leg-${sx}`} position={[sx, -0.8, 0.13]}>
                    <boxGeometry args={[0.06, 0.45, 0.06]} />
                    <meshStandardMaterial color="#a88860" roughness={0.8} />
                  </mesh>
                ))}
                {[-0.55, -0.3, -0.05, 0.2, 0.45].map((bx, bi) => (
                  <mesh key={`book-${bi}`} position={[bx, -0.48 + bi * 0.01, 0]}>
                    <boxGeometry args={[0.16, 0.12 + (bi % 3) * 0.035, 0.22]} />
                    <meshStandardMaterial color={['#4a6670', '#7a6650', '#5a7a60', '#8a6a70', '#607080'][bi]} roughness={0.8} />
                  </mesh>
                ))}
                <mesh position={[0.75, -0.45, 0]}>
                  <cylinderGeometry args={[0.1, 0.08, 0.1, 20]} />
                  <meshStandardMaterial color={COLORS.CELADON_GREEN} roughness={0.25} metalness={0.1} />
                </mesh>
                <mesh position={[0.75, -0.38, 0]}>
                  <cylinderGeometry args={[0.05, 0.035, 0.04, 16]} />
                  <meshStandardMaterial color={COLORS.CELADON_GREEN} roughness={0.25} />
                </mesh>
              </group>
            )}

            {/* 场景2: 经典出海 */}
            {si === 1 && (
              <group position={[0, 0, 0.02]} ref={sceneLightRef}>
                <mesh position={[0, 0.25, 0]}>
                  <sphereGeometry args={[0.55, 48, 48]} />
                  <meshStandardMaterial color="#5a8cb5" roughness={0.5} metalness={0.2} emissive="#3a6080" emissiveIntensity={0.1} />
                </mesh>
                <mesh position={[0, 0.25, -0.02]}>
                  <sphereGeometry args={[0.62, 32, 32]} />
                  <meshBasicMaterial color={COLORS.LIGHT_BLUE_GREEN} transparent opacity={0.12} side={THREE.BackSide} />
                </mesh>
                <mesh position={[0, 0.25, 0]} rotation={[0.3, 0.5, 0.2]}>
                  <ringGeometry args={[0.72, 0.74, 64]} />
                  <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={1} transparent opacity={0.6} side={THREE.DoubleSide} />
                </mesh>
                <mesh position={[0, -0.7, 0]}>
                  <boxGeometry args={[1.6, 0.35, 0.04]} />
                  <meshStandardMaterial color="#ffffff" roughness={0.4} />
                </mesh>
                <Text position={[0, -0.6, 0.03]} fontSize={0.14} color="#c49a5a" anchorX="center">
                  全球180+
                  <meshStandardMaterial color="#c49a5a" emissive="#c49a5a" emissiveIntensity={0.2} />
                </Text>
                <Text position={[0, -0.75, 0.03]} fontSize={0.07} color="#665544" anchorX="center">
                  国家和地区
                  <meshStandardMaterial color="#665544" />
                </Text>
              </group>
            )}

            {/* 场景3: 文化体验 */}
            {si === 2 && (
              <group position={[0, -0.15, 0.02]}>
                <mesh position={[0, -0.5, 0]}>
                  <boxGeometry args={[1.8, 0.45, 0.5]} />
                  <meshStandardMaterial color="#d5c9b8" roughness={0.9} />
                </mesh>
                <mesh position={[0, -0.15, -0.22]}>
                  <boxGeometry args={[1.8, 0.65, 0.08]} />
                  <meshStandardMaterial color="#c8b8a0" roughness={0.9} />
                </mesh>
                <mesh position={[-0.5, -0.3, 0.05]} rotation={[0, 0.1, 0.1]}>
                  <boxGeometry args={[0.3, 0.25, 0.1]} />
                  <meshStandardMaterial color={COLORS.CELADON_GREEN} roughness={0.8} />
                </mesh>
                <mesh position={[0.5, -0.3, 0.05]} rotation={[0, -0.1, -0.1]}>
                  <boxGeometry args={[0.3, 0.25, 0.1]} />
                  <meshStandardMaterial color="#e8c9a0" roughness={0.8} />
                </mesh>
                <mesh position={[0, -0.95, 0.4]}>
                  <boxGeometry args={[1.2, 0.05, 0.4]} />
                  <meshStandardMaterial color="#b5a080" roughness={0.7} />
                </mesh>
                <mesh position={[0.25, -0.85, 0.35]}>
                  <cylinderGeometry args={[0.07, 0.05, 0.15, 16]} />
                  <meshStandardMaterial color={COLORS.CELADON_GREEN} roughness={0.3} metalness={0.1} />
                </mesh>
              </group>
            )}

            {/* 场景标题 */}
            <group position={[0, 1.25, 0.05]}>
              <Text fontSize={0.22} color="#3a3028" anchorX="center" letterSpacing={0.2}>
                {scene.title}
                <meshStandardMaterial color="#3a3028" />
              </Text>
              <mesh position={[0, -0.18, 0]}>
                <planeGeometry args={[1.4, 0.012]} />
                <meshStandardMaterial color={zone.color} emissive={zone.color} emissiveIntensity={0.6} transparent opacity={0.8} />
              </mesh>
              <Text position={[0, -0.35, 0]} fontSize={0.1} color={zone.color} anchorX="center" letterSpacing={0.15}>
                {scene.sub}
                <meshStandardMaterial color={zone.color} />
              </Text>
            </group>
          </group>
        ))}
      </group>
    </ZoneBase>
  )
}

/* ========================================================================
   五大展区主组件
======================================================================== */
export default function Zones() {
  return (
    <group name="exhibition-zones">
      <OriginZone />
      <SurgeZone />
      <RippleZone />
      <ConvergeZone />
      <NourishZone />
    </group>
  )
}
