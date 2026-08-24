import React from 'react'
import { Text } from '@react-three/drei'
import * as THREE from 'three'
import { SPACE, ZONES, ENTRANCE } from '../config/spaceConfig'

/**
 * Debug Mode - 空间尺寸验证工具
 * 显示: 1m网格 / XYZ坐标轴 / 21×13m边界 / 6m高度线 / 1.7m人体参考
 * 默认关闭, 通过UI按钮切换
 */
export default function DebugMode({ visible = false }) {
  if (!visible) return null

  const { WIDTH, DEPTH, HEIGHT, HALF_WIDTH, HALF_DEPTH } = SPACE

  // 生成1m×1m网格线
  const gridLines = []
  // X方向线 (沿Z轴每隔1m)
  for (let x = -HALF_WIDTH; x <= HALF_WIDTH; x += 1) {
    gridLines.push(
      <line key={`grid-x-${x}`} position={[x, 0.01, 0]}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, -HALF_DEPTH, 0, 0, HALF_DEPTH])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={x === 0 ? '#ff4444' : '#444444'} transparent opacity={0.3} />
      </line>
    )
  }
  // Z方向线 (沿X轴每隔1m)
  for (let z = -HALF_DEPTH; z <= HALF_DEPTH; z += 1) {
    gridLines.push(
      <line key={`grid-z-${z}`} position={[0, 0.01, z]}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([-HALF_WIDTH, 0, 0, HALF_WIDTH, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color={z === 0 ? '#ff4444' : '#444444'} transparent opacity={0.3} />
      </line>
    )
  }

  return (
    <group name="debug-mode">
      {/* === 1m×1m 网格 === */}
      {gridLines}

      {/* === 21m×13m 场地边界 (红色粗线) === */}
      <line position={[0, 0.02, 0]}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={5}
            array={new Float32Array([
              -HALF_WIDTH, 0, -HALF_DEPTH,
              HALF_WIDTH, 0, -HALF_DEPTH,
              HALF_WIDTH, 0, HALF_DEPTH,
              -HALF_WIDTH, 0, HALF_DEPTH,
              -HALF_WIDTH, 0, -HALF_DEPTH,
            ])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#ff0000" linewidth={2} />
      </line>

      {/* === 6m 高度参考线 (四个角) === */}
      {[
        [-HALF_WIDTH, -HALF_DEPTH],
        [HALF_WIDTH, -HALF_DEPTH],
        [HALF_WIDTH, HALF_DEPTH],
        [-HALF_WIDTH, HALF_DEPTH],
      ].map(([x, z], i) => (
        <group key={`height-ref-${i}`}>
          <line position={[x, 0, z]}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([0, 0, 0, 0, HEIGHT, 0])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#ff8800" transparent opacity={0.6} />
          </line>
          <Text
            position={[0, HEIGHT + 0.2, 0]}
            fontSize={0.3}
            color="#ff8800"
            anchorX="center"
          >
            6m
            <meshBasicMaterial color="#ff8800" />
          </Text>
        </group>
      ))}

      {/* === 1.7m 人体尺度参考线 === */}
      {[
        [-4, 0],
        [4, 0],
        [0, -3],
        [0, 3],
      ].map(([x, z], i) => (
        <group key={`human-ref-${i}`} position={[x, 0, z]}>
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([0, 0, 0, 0, 1.7, 0])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#00ff00" />
          </line>
          <Text
            position={[0.3, 1.7, 0]}
            fontSize={0.2}
            color="#00ff00"
            anchorX="left"
          >
            1.7m
            <meshBasicMaterial color="#00ff00" />
          </Text>
        </group>
      ))}

      {/* === 中央装置尺寸标注 === */}
      {/* 底部直径4m (半径2m) */}
      <line position={[0, 0.03, 0]}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([-2, 0, 0, 2, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#00ffff" />
      </line>
      <Text
        position={[0, 0.1, 0]}
        fontSize={0.2}
        color="#00ffff"
        anchorX="center"
      >
        装置底径4m
        <meshBasicMaterial color="#00ffff" />
      </Text>
      {/* 装置高度4.5m */}
      <line position={[2.2, 0, 0]}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, 4.5, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#00ffff" />
      </line>
      <Text
        position={[2.5, 2.25, 0]}
        fontSize={0.2}
        color="#00ffff"
        anchorX="left"
      >
        装置高4.5m
        <meshBasicMaterial color="#00ffff" />
      </Text>

      {/* === 展墙高度标注 (2.8m) === */}
      {[
        { x: ZONES.ORIGIN.position.x, z: ZONES.ORIGIN.position.z, label: '展墙2.8m' },
        { x: ZONES.SURGE.position.x, z: ZONES.SURGE.position.z, label: '展墙2.8m' },
      ].map((item, i) => (
        <group key={`wall-ref-${i}`} position={[item.x - 3, 0, item.z]}>
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([0, 0, 0, 0, 2.8, 0])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#ff44ff" />
          </line>
          <Text
            position={[0.3, 2.8, 0]}
            fontSize={0.18}
            color="#ff44ff"
            anchorX="left"
          >
            {item.label}
            <meshBasicMaterial color="#ff44ff" />
          </Text>
        </group>
      ))}

      {/* === 门头高度标注 (4.2m) === */}
      <group position={[ENTRANCE.position.x, 0, ENTRANCE.position.z]}>
        <line position={[3, 0, 0]}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([0, 0, 0, 0, 4.2, 0])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#ffaa00" />
        </line>
        <Text
          position={[3.3, 4.2, 0]}
          fontSize={0.18}
          color="#ffaa00"
          anchorX="left"
        >
          门头4.2m
          <meshBasicMaterial color="#ffaa00" />
        </Text>
      </group>

      {/* === 通道宽度标注 === */}
      {/* 主通道宽度 (中央装置边缘到展区) */}
      <line position={[2.5, 0.05, 0]}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 4, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#ffff00" />
      </line>
      <Text
        position={[4.5, 0.15, 0]}
        fontSize={0.18}
        color="#ffff00"
        anchorX="left"
      >
        通道≈4m
        <meshBasicMaterial color="#ffff00" />
      </Text>

      {/* === 入口净宽标注 === */}
      <line position={[ENTRANCE.position.x, 0.05, ENTRANCE.position.z + 0.5]}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([-ENTRANCE.width / 2, 0, 0, ENTRANCE.width / 2, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#88ff00" />
      </line>
      <Text
        position={[0, 0.2, ENTRANCE.position.z + 0.5]}
        fontSize={0.2}
        color="#88ff00"
        anchorX="center"
      >
        入口净宽5m
        <meshBasicMaterial color="#88ff00" />
      </Text>

      {/* === 展馆总尺寸标注 === */}
      <Text
        position={[0, 0.1, -HALF_DEPTH - 0.5]}
        fontSize={0.3}
        color="#ff0000"
        anchorX="center"
        rotation={[-Math.PI / 2, 0, 0]}
      >
        21m
        <meshBasicMaterial color="#ff0000" />
      </Text>
      <Text
        position={[-HALF_WIDTH - 0.5, 0.1, 0]}
        fontSize={0.3}
        color="#ff0000"
        anchorX="center"
        rotation={[-Math.PI / 2, 0, Math.PI / 2]}
      >
        13m
        <meshBasicMaterial color="#ff0000" />
      </Text>
    </group>
  )
}
