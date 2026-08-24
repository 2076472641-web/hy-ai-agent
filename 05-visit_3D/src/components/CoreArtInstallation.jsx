import React, { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { COLORS, ZONES } from '../config/spaceConfig'

/* ========================================================================
   中央艺术装置 - 潮生之芯 TIDAL CORE  v2.0
   设计概念：钱塘潮 · 丝绸飘 · 青瓷韵 · 数据流
   高度4.5m  底部直径4m  中腰收束  顶部绽放
   "水流向天空生长" 的有机结构 - 东方水墨丝绸流动感
========================================================================
   性能策略：
   - 32条曲线全部使用 ShaderMaterial (移除昂贵的 transmission)
   - 粒子总量 720 (500 流动 + 220 地图)
   - 水波 10 层 (减半 mesh)
   - 流动粒子分批 1/3 更新
   - 整体慢速呼吸 (180s/圈)
======================================================================== */

// ========== 浙江地图简化轮廓 (归一化极坐标 r∈[0,1], θ∈[0,2π]) ==========
// 顺时针闭合多边形 - 北窄(湖州)/杭州湾内凹/南宽(温州)/西凸(衢州丽水)
const ZHEJIANG_PROFILE = [
  [0.62, 0.10], [0.58, 0.45], [0.70, 0.80], [0.78, 1.15],
  [0.82, 1.50], [0.95, 1.85], [1.00, 2.20], [0.90, 2.55],
  [0.85, 2.90], [0.92, 3.25], [0.88, 3.60], [0.75, 3.95],
  [0.62, 4.30], [0.55, 4.65], [0.50, 5.00], [0.45, 5.35],
  [0.52, 5.70], [0.68, 6.05], [0.72, 6.18],
]

// ========== 简易伪噪声 (替代纯sin, 产生更自然的水墨流动) ==========
const hash = (n) => {
  const s = Math.sin(n) * 43758.5453
  return s - Math.floor(s)
}
// 1D 平滑噪声 - 用于曲线扰动
const noise1D = (x) => {
  const i = Math.floor(x)
  const f = x - i
  const u = f * f * (3 - 2 * f)
  return hash(i) * (1 - u) + hash(i + 1) * u
}
// 分形噪声 - 多频率叠加
const fbm = (x) => {
  return noise1D(x) * 0.6 + noise1D(x * 2.1) * 0.3 + noise1D(x * 4.3) * 0.1
}

export default function CoreArtInstallation() {
  const groupRef = useRef()
  const tubeRefs = useRef([])
  const tubeUnis = useRef([])
  const centerGlowRef = useRef()
  const centerGlowHaloRef = useRef()
  const flowParticlesRef = useRef()
  const mapParticlesRef = useRef()
  const rippleMeshRefs = useRef([])
  const haloRing1Ref = useRef()
  const haloRing2Ref = useRef()
  const haloBaseRef = useRef()
  const tideWaveRef = useRef()
  const topCrownRef = useRef()

  // ============================================================
  // 1. 曲线生成系统 - 32条 3层分布 (潮涌 / 丝绸 / 绽放)
  // ============================================================
  const TOTAL_CURVES = 32
  const HEIGHT = 4.5

  // 半径轮廓函数 r(t): 底展 → 中束 → 顶展
  // t∈[0,1]，y=0→4.5m
  // 底部直径4m(r=2.0) → 中部直径≈0.9m(r=0.45) → 顶部3.2m(r=1.6)
  const radiusProfile = (t) => {
    if (t < 0.38) {
      // 从底部到收束点 平滑收缩 (smoothstep)
      const s = t / 0.38
      const ease = s * s * (3 - 2 * s)
      return 2.0 - (2.0 - 0.48) * ease
    } else if (t < 0.72) {
      // 收束段 - 微微凹陷，如水壶腰
      const s = (t - 0.38) / 0.34
      return 0.48 + Math.sin(s * Math.PI) * -0.06
    } else {
      // 顶部绽放 - 加速向外展开 (cubic ease out)
      const s = (t - 0.72) / 0.28
      const ease = s * s * (3 - 2 * s)  // smoothstep 更柔和
      return 0.42 + (1.6 - 0.42) * ease
    }
  }

  // 生成单条有机曲线 - 分3层不同风格
  const generateTidalCurve = (i, total) => {
    const points = []
    const N_PNTS = 70  // 采样点 (CatmullRom会平滑)

    // 每条曲线独特种子 (用 hash 替代 Math.random 更稳定)
    const seed = i * 13.37
    const angleOffset = (i / total) * Math.PI * 2
    const phaseA = hash(seed + 1.1) * Math.PI * 2
    const phaseB = hash(seed + 2.2) * Math.PI * 2
    const phaseC = hash(seed + 3.3) * Math.PI * 2

    // 分层判定
    let layer, totalTurns, wavAmp, wavFreq, rScale, verticalSway
    if (i < 9) {
      // === 外层 潮涌 (tide) ===
      // 大弧度旋转上升,如钱塘潮涌
      layer = 'tide'
      totalTurns = 0.55 + hash(seed) * 0.55   // 0.55~1.1 圈 (少而大气)
      wavAmp = 0.10 + (i % 3) * 0.022
      wavFreq = 1.6 + hash(seed + 4) * 0.6
      rScale = 1.08 + (i % 3) * 0.04           // 略外凸
      verticalSway = 0.04                       // 轻微垂直摆动
    } else if (i < 24) {
      // === 中层 丝绸 (silk) ===
      // 多圈细密旋转,如丝绸飘带
      layer = 'silk'
      totalTurns = 1.1 + hash(seed) * 1.4      // 1.1~2.5 圈 (飘逸)
      wavAmp = 0.045 + (i % 5) * 0.012
      wavFreq = 3.8 + hash(seed + 4) * 1.2
      rScale = 0.92 + (i % 5) * 0.025
      verticalSway = 0.06                       // 丝绸飘动幅度较大
    } else {
      // === 顶层 绽放 (bloom) ===
      // 顶部向外绽放,如青瓷花瓣
      layer = 'bloom'
      totalTurns = 0.4 + hash(seed) * 0.5      // 0.4~0.9 圈 (少,显开瓣)
      wavAmp = 0.06 + ((i - 24) % 4) * 0.015
      wavFreq = 2.5 + hash(seed + 4) * 0.8
      rScale = 0.82 + ((i - 24) % 4) * 0.03
      verticalSway = 0.03
    }

    for (let j = 0; j <= N_PNTS; j++) {
      const t = j / N_PNTS
      const r = radiusProfile(t) * rScale

      // 主旋转角 - 围绕中央轴旋转上升
      const mainAngle = angleOffset + t * Math.PI * 2 * totalTurns

      // === 多频率分形扰动 (取代单sin, 更有机) ===
      // 1. 主水波 (沿曲线方向流动)
      const wave1 = Math.sin(t * Math.PI * wavFreq + phaseA) * wavAmp
      // 2. 次水波 (倍频, 制造褶皱细节)
      const wave2 = Math.sin(t * Math.PI * wavFreq * 2.1 + phaseB) * wavAmp * 0.42
      // 3. fbm 噪声扰动 (有机水墨晕染感)
      const wave3 = (fbm(t * 4 + seed * 0.3) - 0.5) * wavAmp * 1.6
      // 4. 低频横向漂移 (丝绸整体飘动)
      const wave4 = Math.cos(t * Math.PI * 0.7 + phaseC) * wavAmp * 0.3
      const perturb = wave1 + wave2 + wave3 + wave4

      // 扰动半径 - 产生丝绸褶皱感
      const localR = Math.max(0.05, r + perturb)

      // === 顶部绽放额外径向分量 - 各方向强度不同 ===
      let topSpread = 0
      if (t > 0.68) {
        const s = (t - 0.68) / 0.32
        const spreadFactor = s * s
        const dirRad = 0.32 + ((i * 37) % 100) / 150
        topSpread = Math.cos(t * 5 + i) * spreadFactor * dirRad
      }

      // === 底部根部分散 - 不集中成束 ===
      let baseSpread = 0
      if (t < 0.08) {
        baseSpread = (0.08 - t) * 0.8 * Math.sin(phaseA + i)
      }

      const finalR = Math.max(0.04, localR + topSpread + baseSpread)
      const x = Math.cos(mainAngle) * finalR
      const z = Math.sin(mainAngle) * finalR

      // 垂直摆动 (丝绸/水墨流动感)
      const yWobble = Math.sin(t * Math.PI * 1.8 + phaseB) * verticalSway * 0.4
      const y = 0.08 + t * HEIGHT + yWobble

      points.push(new THREE.Vector3(x, y, z))
    }

    return new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5)
  }

  // 曲线数据预处理 (useMemo 一次性构建)
  // 同时预计算 LUT (查找表) 用于粒子快速定位 - 避免每帧 curve.getPoint() 开销
  const LUT_SIZE = 200  // 每条曲线预采样 200 点
  const { curves, tubeMetas, curveLUTs } = useMemo(() => {
    const curvesArr = []
    const metasArr = []
    const lutsArr = []  // 每条曲线的 LUT: Float32Array(LUT_SIZE * 3)

    for (let i = 0; i < TOTAL_CURVES; i++) {
      const curve = generateTidalCurve(i, TOTAL_CURVES)
      curvesArr.push(curve)

      // 预计算 LUT - 一次性采样, useFrame 用线性插值查询
      const lut = new Float32Array(LUT_SIZE * 3)
      for (let k = 0; k < LUT_SIZE; k++) {
        const p = curve.getPoint(k / (LUT_SIZE - 1))
        lut[k * 3] = p.x
        lut[k * 3 + 1] = p.y
        lut[k * 3 + 2] = p.z
      }
      lutsArr.push(lut)

      let style
      if (i < 9) style = 'tide'
      else if (i < 24) style = 'silk'
      else style = 'bloom'

      // === tube 半径 - 沿长度变化 (丝绸根粗梢细) ===
      // 用函数表达,但 TubeGeometry 只能用单一半径,所以用平均值+shader控制
      let tubeRadius
      if (style === 'tide') {
        tubeRadius = 0.04 + (i % 3) * 0.008       // 粗 - 潮涌有分量
      } else if (style === 'silk') {
        tubeRadius = 0.022 + (i % 5) * 0.005      // 细 - 丝绸飘带
      } else {
        tubeRadius = 0.03 + ((i - 24) % 4) * 0.006
      }

      // === 颜色分层 (青瓷→淡蓝绿→瓷白) ===
      const grad = i / (TOTAL_CURVES - 1)
      let color
      if (grad < 0.28) {
        // 底层潮涌 - 深青瓷
        const s = grad / 0.28
        color = new THREE.Color(COLORS.CELADON_GREEN).lerp(new THREE.Color('#5a9489'), s)
      } else if (grad < 0.72) {
        // 中层丝绸 - 青绿过渡
        const s = (grad - 0.28) / 0.44
        color = new THREE.Color('#5a9489').lerp(new THREE.Color(COLORS.LIGHT_BLUE_GREEN), s)
      } else {
        // 顶层绽放 - 瓷白润青
        const s = (grad - 0.72) / 0.28
        color = new THREE.Color(COLORS.LIGHT_BLUE_GREEN).lerp(new THREE.Color('#e8f3f1'), s)
      }

      // === 透明度 / 发光强度分层 ===
      let opacityMul, glowStrength, flowSpeed
      if (style === 'tide') {
        opacityMul = 0.55 + (i % 3) * 0.04
        glowStrength = 0.55
        flowSpeed = 0.9
      } else if (style === 'silk') {
        opacityMul = 0.42 + (i % 5) * 0.035
        glowStrength = 0.75   // 丝绸高亮
        flowSpeed = 1.2
      } else {
        opacityMul = 0.38 + ((i - 24) % 4) * 0.04
        glowStrength = 0.65
        flowSpeed = 0.7
      }

      metasArr.push({
        style, tubeRadius, color, opacityMul, glowStrength, flowSpeed, index: i
      })
    }
    return { curves: curvesArr, tubeMetas: metasArr, curveLUTs: lutsArr }
  }, [])

  // ============================================================
  // 2. 流动粒子系统 (500个，沿32条曲线向上爬升)
  // ============================================================
  const flowParticles = useMemo(() => {
    const COUNT = 500
    const positions = new Float32Array(COUNT * 3)
    const sizes = new Float32Array(COUNT)
    const progArray = new Float32Array(COUNT)  // 进度 t∈[0,1]
    const curveIdx = new Uint8Array(COUNT)
    const speeds = new Float32Array(COUNT)
    const phases = new Float32Array(COUNT)

    for (let i = 0; i < COUNT; i++) {
      // 按曲线权重分配 (silk 层多分一些,视觉密度)
      let cIdx
      const r = Math.random()
      if (r < 0.25) cIdx = Math.floor(Math.random() * 9)              // tide 25%
      else if (r < 0.75) cIdx = 9 + Math.floor(Math.random() * 15)    // silk 50%
      else cIdx = 24 + Math.floor(Math.random() * 8)                  // bloom 25%
      curveIdx[i] = cIdx

      progArray[i] = Math.random()
      // 速度梯度 - 越往上越慢 (诗意飘浮)
      speeds[i] = 0.012 + Math.random() * 0.010
      positions[i * 3] = 0
      positions[i * 3 + 1] = 0.2
      positions[i * 3 + 2] = 0
      sizes[i] = 0.025 + Math.random() * 0.035
      phases[i] = Math.random() * Math.PI * 2
    }

    return { count: COUNT, positions, sizes, progArray, curveIdx, speeds, phases }
  }, [])

  // ============================================================
  // 3. 顶部浙江地图轮廓粒子 (220个 - 更聚焦轮廓边缘)
  // ============================================================
  const mapParticles = useMemo(() => {
    const COUNT = 220
    const positions = new Float32Array(COUNT * 3)
    const targets = new Float32Array(COUNT * 3)
    const sizes = new Float32Array(COUNT)
    const lerpRates = new Float32Array(COUNT)
    const phases = new Float32Array(COUNT)

    // 边界点数 / 内部点数 比例 - 60% 边缘清晰, 40% 内部填充
    const edgeCount = Math.floor(COUNT * 0.6)
    const profileLen = ZHEJIANG_PROFILE.length

    for (let i = 0; i < COUNT; i++) {
      let r, theta
      if (i < edgeCount) {
        // === 边缘点 - 沿浙江轮廓多边形采样 ===
        const segIdx = Math.floor(Math.random() * profileLen)
        const nextIdx = (segIdx + 1) % profileLen
        const p1 = ZHEJIANG_PROFILE[segIdx]
        const p2 = ZHEJIANG_PROFILE[nextIdx]
        const segT = Math.random()
        r = p1[0] + (p2[0] - p1[0]) * segT
        theta = p1[1] + (p2[1] - p1[1]) * segT
        // 边缘微小抖动 - 手绘水墨感
        r += (Math.random() - 0.5) * 0.03
        theta += (Math.random() - 0.5) * 0.04
      } else {
        // === 内部填充点 - 径向插值 ===
        const profileIdx = Math.floor(Math.random() * profileLen)
        const nextIdx = (profileIdx + 1) % profileLen
        const p1 = ZHEJIANG_PROFILE[profileIdx]
        const p2 = ZHEJIANG_PROFILE[nextIdx]
        const segT = Math.random()
        const edgeR = p1[0] + (p2[0] - p1[0]) * segT
        const edgeTheta = p1[1] + (p2[1] - p1[1]) * segT
        // 内部填充 sqrt 分布
        const inner = Math.sqrt(Math.random()) * 0.85
        r = edgeR * inner
        theta = edgeTheta + (Math.random() - 0.5) * 0.06
      }

      // 地图投影在装置顶部,中心对准,半径约1.4m
      const MAP_SCALE = 1.4
      const MAP_Y_MIN = 3.85
      const MAP_Y_MAX = 4.55
      const tx = Math.cos(theta) * r * MAP_SCALE
      const tz = Math.sin(theta) * r * MAP_SCALE
      const ty = MAP_Y_MIN + Math.random() * (MAP_Y_MAX - MAP_Y_MIN)

      targets[i * 3] = tx
      targets[i * 3 + 1] = ty
      targets[i * 3 + 2] = tz

      // 初始散乱位置
      positions[i * 3] = (Math.random() - 0.5) * 4.5
      positions[i * 3 + 1] = 3.5 + Math.random() * 1.4
      positions[i * 3 + 2] = (Math.random() - 0.5) * 4.5

      sizes[i] = 0.03 + Math.random() * 0.035
      lerpRates[i] = 0.0025 + Math.random() * 0.006  // 极慢收敛
      phases[i] = Math.random() * Math.PI * 2
    }

    return { count: COUNT, positions, targets, sizes, lerpRates, phases }
  }, [])

  // ============================================================
  // 4. 主动画帧 - 整体慢速诗意
  // ============================================================
  useFrame((state, delta) => {
    const t = state.clock.elapsedTime
    const dt = Math.min(delta, 0.05)

    // ------- 4.1 整体缓慢旋转 (180秒/圈,极慢,安静) + 垂直呼吸 -------
    if (groupRef.current) {
      groupRef.current.rotation.y = t * 0.0349   // 2π/180 ≈ 0.0349
      // 整体微弱呼吸缩放 (0.5% 幅度, 肉眼几乎不可察的诗意感)
      const breath = 1 + Math.sin(t * 0.28) * 0.006
      groupRef.current.scale.setScalar(breath)
    }

    // ------- 4.2 tube shader uniforms 更新 (流动光纹) -------
    for (let i = 0; i < TOTAL_CURVES; i++) {
      const u = tubeUnis.current[i]
      if (u) {
        u.time.value = t * 0.18
        u.pulse.value = 0.88 + Math.sin(t * 0.35 + i * 0.22) * 0.12
      }
    }

    // ------- 4.3 中央发光核心脉动 -------
    if (centerGlowRef.current) {
      const pulse = 1 + Math.sin(t * 0.32) * 0.05
      centerGlowRef.current.scale.setScalar(pulse)
      const mat = centerGlowRef.current.material
      mat.emissiveIntensity = 1.2 + Math.sin(t * 0.32) * 0.3
    }
    if (centerGlowHaloRef.current) {
      const breath = 1 + Math.sin(t * 0.32 + 0.6) * 0.08
      centerGlowHaloRef.current.scale.setScalar(breath)
      centerGlowHaloRef.current.material.opacity = 0.10 + Math.sin(t * 0.32) * 0.04
    }

    // ------- 4.4 流动粒子沿曲线爬升 (分批1/3, 用LUT替代getPoint) -------
    if (flowParticlesRef.current) {
      const posArr = flowParticlesRef.current.geometry.attributes.position.array
      const { progArray, curveIdx, speeds, phases, count } = flowParticles

      const batchSize = Math.ceil(count / 3)
      const batchOffset = Math.floor(t / 0.12) % 3

      for (let i = batchOffset * batchSize; i < Math.min(count, (batchOffset + 1) * batchSize); i++) {
        // 推进进度 (越接近顶部越慢 - 诗意漂浮)
        const prog = progArray[i]
        const slowDown = 1.0 - prog * 0.35
        let p = prog + speeds[i] * dt * slowDown * (1.0 + 0.15 * Math.sin(t * 0.2 + phases[i]))
        // 到达顶部后从底部重置 (带随机偏移)
        if (p > 1.02) p = -0.02 - Math.random() * 0.04
        if (p < 0) p = Math.max(0, p + 1.0)
        progArray[i] = p

        // === LUT 快速查询 (替代 curve.getPoint, 提速5-10倍) ===
        const lut = curveLUTs[curveIdx[i]]
        if (lut) {
          const clampedP = p < 0 ? 0 : (p > 1 ? 1 : p)
          const fIdx = clampedP * (LUT_SIZE - 1)
          const i0 = Math.floor(fIdx)
          const i1 = Math.min(LUT_SIZE - 1, i0 + 1)
          const frac = fIdx - i0
          const px = lut[i0 * 3]     + (lut[i1 * 3]     - lut[i0 * 3])     * frac
          const py = lut[i0 * 3 + 1] + (lut[i1 * 3 + 1] - lut[i0 * 3 + 1]) * frac
          const pz = lut[i0 * 3 + 2] + (lut[i1 * 3 + 2] - lut[i0 * 3 + 2]) * frac
          // 轻微浮动 - 丝绸飘动
          const floatR = 0.015 + Math.sin(t * 0.8 + phases[i]) * 0.012
          const floatAng = t * 0.5 + phases[i]
          posArr[i * 3]     = px + Math.cos(floatAng) * floatR
          posArr[i * 3 + 1] = py + Math.sin(t * 0.6 + phases[i] * 1.3) * 0.015
          posArr[i * 3 + 2] = pz + Math.sin(floatAng) * floatR
        }
      }
      flowParticlesRef.current.geometry.attributes.position.needsUpdate = true
    }

    // ------- 4.5 顶部浙江轮廓粒子缓慢汇聚 + 浮动 (分批1/2) -------
    if (mapParticlesRef.current) {
      const posArr = mapParticlesRef.current.geometry.attributes.position.array
      const { targets, lerpRates, phases, count } = mapParticles

      // 分批更新 - 每帧处理一半粒子, 肉眼无差异
      const halfCount = Math.ceil(count / 2)
      const batchOffset = Math.floor(t / 0.16) % 2
      const startI = batchOffset * halfCount
      const endI = Math.min(count, startI + halfCount)

      for (let i = startI; i < endI; i++) {
        const ix = i * 3, iy = i * 3 + 1, iz = i * 3 + 2
        // Lerp 趋向目标
        posArr[ix] += (targets[ix] - posArr[ix]) * lerpRates[i]
        posArr[iy] += (targets[iy] - posArr[iy]) * lerpRates[i]
        posArr[iz] += (targets[iz] - posArr[iz]) * lerpRates[i]
        // 浮动呼吸 (慢,小幅)
        posArr[ix] += Math.sin(t * 0.45 + phases[i]) * 0.0025
        posArr[iy] += Math.sin(t * 0.38 + phases[i] * 1.2) * 0.004
        posArr[iz] += Math.cos(t * 0.42 + phases[i] * 0.9) * 0.0025
      }
      mapParticlesRef.current.geometry.attributes.position.needsUpdate = true
    }

    // ------- 4.6 地面水墨水波扩散 (10层,周期5秒,更慢更诗意) -------
    const RIPPLE_COUNT = 10
    for (let i = 0; i < RIPPLE_COUNT; i++) {
      const ripple = rippleMeshRefs.current[i]
      if (ripple) {
        const period = 5.0
        const phase = ((t / period) + i / RIPPLE_COUNT) % 1.0
        // scale 0.05 → 1.0 扩散
        ripple.scale.setScalar(0.04 + phase * 0.96)
        // 中间清晰,边缘水墨晕染消散
        const inkOpacity = Math.max(0, 0.45 - Math.abs(phase - 0.5) * 0.8)
        ripple.material.opacity = inkOpacity
      }
    }

    // ------- 4.7 主光环呼吸 + 旋转 (慢) -------
    if (haloRing1Ref.current) {
      const breath = 0.82 + Math.sin(t * 0.65) * 0.18
      haloRing1Ref.current.material.opacity = 0.72 * breath
      haloRing1Ref.current.rotation.z = t * 0.12
    }
    if (haloRing2Ref.current) {
      const breath = 0.88 + Math.sin(t * 0.65 + 0.8) * 0.12
      haloRing2Ref.current.material.opacity = 0.55 * breath
      haloRing2Ref.current.rotation.z = -t * 0.09
    }
    if (haloBaseRef.current) {
      const breath = 0.9 + Math.sin(t * 0.4) * 0.1
      haloBaseRef.current.material.opacity = 0.55 * breath
    }
    if (tideWaveRef.current) {
      tideWaveRef.current.rotation.z = t * 0.06
      const pulse = 0.85 + Math.sin(t * 0.55) * 0.15
      tideWaveRef.current.material.opacity = 0.42 * pulse
    }
    if (topCrownRef.current) {
      topCrownRef.current.rotation.y = t * 0.15
    }
  })

  // ============================================================
  // 5. 通用 Tube Shader - 丝绸流动 / 水墨晕染 / 数据脉
  // (性能: 全部 ShaderMaterial 替代 MeshPhysicalMaterial+transmission)
  // ============================================================
  const createTubeShader = (meta) => {
    const baseColor = meta.color.clone()
    const accentColor = new THREE.Color(COLORS.LIGHT_BLUE_GREEN)
    const porcelainColor = new THREE.Color('#e8f3f1')
    return {
      uniforms: {
        time:      { value: 0 },
        pulse:     { value: 1.0 },
        baseC:     { value: baseColor },
        accent:    { value: accentColor },
        porcelain: { value: porcelainColor },
        opacityMul:    { value: meta.opacityMul },
        glowStrength:  { value: meta.glowStrength },
        flowSpeed:     { value: meta.flowSpeed },
      },
      vertexShader: `
        varying vec2 vUv;
        varying float vY;
        void main() {
          vUv = uv;
          vY = position.y;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform float pulse;
        uniform vec3 baseC;
        uniform vec3 accent;
        uniform vec3 porcelain;
        uniform float opacityMul;
        uniform float glowStrength;
        uniform float flowSpeed;
        varying vec2 vUv;
        varying float vY;

        void main() {
          // === 1. 沿长度方向流动亮纹 (丝绸/数据脉) ===
          float alongU = vUv.x;   // 曲线长度方向 0~1
          float acrossV = vUv.y;  // 横截面 0~1

          // 主流动条纹 (慢,数据脉)
          float stripes = sin(alongU * 50.0 - time * 3.0 * flowSpeed) * 0.5 + 0.5;
          // 细密高频条纹 (丝绸经纬)
          float fineStripes = sin(alongU * 180.0 - time * 5.0 * flowSpeed + stripes * 1.5) * 0.5 + 0.5;

          // === 2. 高度色阶 (底深 → 中亮 → 顶瓷白) ===
          float heightT = clamp(vY / 4.5, 0.0, 1.0);
          // 中部偏上亮一点 (数据汇聚点)
          float glowZone = smoothstep(0.30, 0.55, heightT) * (1.0 - smoothstep(0.55, 0.80, heightT));

          // === 3. 颜色混合 ===
          vec3 col = mix(baseC, accent, stripes * 0.30 + glowZone * 0.40);
          // 顶部加入瓷白润色 (青瓷釉色) - 用 heightT*heightT 替代 pow 优化
          float porcelainMix = heightT * heightT * 0.55;
          col = mix(col, porcelain, porcelainMix);

          // === 4. 透明度: 横截面边缘羽化 + 头尾淡入淡出 ===
          float edge = smoothstep(0.0, 0.18, acrossV) * (1.0 - smoothstep(0.82, 1.0, acrossV));
          float tipFade = smoothstep(0.0, 0.05, alongU) * (1.0 - smoothstep(0.94, 1.0, alongU));
          float alpha = edge * tipFade * opacityMul * pulse;

          // === 5. 发光 (丝绸高光 + 数据脉) ===
          float glow = stripes * 0.32 + fineStripes * 0.10 + glowZone * 0.55;
          vec3 finalCol = col * (0.72 + glow * glowStrength);

          gl_FragColor = vec4(finalCol, clamp(alpha, 0.0, 0.88));
        }
      `,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: true,
      blending: THREE.NormalBlending,
    }
  }

  // ============================================================
  // 6. 渲染
  // ============================================================
  return (
    <group name="tidal-core" position={[ZONES.CENTER.x, 0, ZONES.CENTER.z]}>
      <group ref={groupRef}>

        {/* ============================================================
            中央发光核心 - 青瓷润心 (高度≈2.0m处)
        ============================================================ */}
        <mesh ref={centerGlowRef} position={[0, 2.0, 0]}>
          <icosahedronGeometry args={[0.22, 3]} />
          <meshStandardMaterial
            color={COLORS.LIGHT_BLUE_GREEN}
            emissive={COLORS.CELADON_GREEN}
            emissiveIntensity={1.3}
            transparent
            opacity={0.92}
            roughness={0.08}
            metalness={0.35}
          />
        </mesh>
        {/* 核心柔光晕 (青瓷光晕) */}
        <mesh ref={centerGlowHaloRef} position={[0, 2.0, 0]}>
          <sphereGeometry args={[0.48, 32, 32]} />
          <meshBasicMaterial
            color={COLORS.CELADON_GREEN}
            transparent
            opacity={0.10}
            side={THREE.BackSide}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
        {/* 极细丝线从核心发射 - 视觉中心锚点 */}
        {[...Array(14)].map((_, k) => {
          const a = (k / 14) * Math.PI * 2
          return (
            <mesh key={`aura-${k}`} position={[Math.cos(a) * 0.04, 2.0, Math.sin(a) * 0.04]} rotation={[0, a, Math.PI / 2.3]}>
              <cylinderGeometry args={[0.003, 0.001, 0.6, 4, 1, true]} />
              <meshBasicMaterial
                color={COLORS.LIGHT_BLUE_GREEN}
                transparent
                opacity={0.32}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
              />
            </mesh>
          )
        })}

        {/* ============================================================
            32条 3层有机曲线 Tube (全部 ShaderMaterial)
            renderOrder 分层确保透明排序: tide=1, bloom=2, silk=3
        ============================================================ */}
        {curves.map((curve, i) => {
          const meta = tubeMetas[i]
          const shader = createTubeShader(meta)
          // 不同层不同细分 - 性能优化 (Intel UHD 集显友好)
          const tubular = meta.style === 'silk' ? 110 : (meta.style === 'tide' ? 90 : 80)
          const radial  = 6
          return (
            <mesh
              key={`tube-${i}`}
              ref={(el) => {
                tubeRefs.current[i] = el
                if (el) tubeUnis.current[i] = el.material.uniforms
              }}
              renderOrder={meta.style === 'silk' ? 3 : (meta.style === 'tide' ? 1 : 2)}
            >
              <tubeGeometry args={[curve, tubular, meta.tubeRadius, radial, false]} />
              <shaderMaterial attach="material" args={[shader]} />
            </mesh>
          )
        })}

        {/* ============================================================
            沿曲线流动粒子 - 蓝绿色数字数据流 (500个)
        ============================================================ */}
        <points ref={flowParticlesRef} renderOrder={4}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" count={flowParticles.count}
              array={flowParticles.positions} itemSize={3} />
            <bufferAttribute attach="attributes-size" count={flowParticles.count}
              array={flowParticles.sizes} itemSize={1} />
          </bufferGeometry>
          <pointsMaterial
            color={COLORS.LIGHT_BLUE_GREEN}
            size={0.04}
            transparent
            opacity={0.9}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            sizeAttenuation
          />
        </points>

        {/* ============================================================
            顶部浙江地图轮廓粒子 - 抽象汇聚 (220个)
        ============================================================ */}
        <points ref={mapParticlesRef} renderOrder={5}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" count={mapParticles.count}
              array={mapParticles.positions} itemSize={3} />
            <bufferAttribute attach="attributes-size" count={mapParticles.count}
              array={mapParticles.sizes} itemSize={1} />
          </bufferGeometry>
          <pointsMaterial
            color="#c6e5df"
            size={0.048}
            transparent
            opacity={0.82}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            sizeAttenuation
          />
        </points>

        {/* ============================================================
            顶端青瓷润白焦点 - 最高点
        ============================================================ */}
        <group ref={topCrownRef} position={[0, 4.58, 0]}>
          <mesh>
            <cylinderGeometry args={[0.22, 0.06, 0.16, 28, 1, true]} />
            <meshPhysicalMaterial
              color={COLORS.CELADON_GREEN}
              transparent
              opacity={0.55}
              transmission={0.4}
              thickness={0.2}
              clearcoat={1.0}
              clearcoatRoughness={0.06}
              side={THREE.DoubleSide}
              roughness={0.12}
              emissive={COLORS.CELADON_GREEN}
              emissiveIntensity={0.3}
            />
          </mesh>
          <mesh position={[0, 0.12, 0]}>
            <sphereGeometry args={[0.05, 18, 18]} />
            <meshStandardMaterial
              color="#ffffff"
              emissive={COLORS.WARM_LIGHT}
              emissiveIntensity={1.6}
            />
          </mesh>
          {/* 顶端柔光 */}
          <mesh position={[0, 0.12, 0]}>
            <sphereGeometry args={[0.18, 20, 20]} />
            <meshBasicMaterial
              color={COLORS.LIGHT_BLUE_GREEN}
              transparent
              opacity={0.22}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
              side={THREE.BackSide}
            />
          </mesh>
        </group>

      </group>

      {/* ============================================================
          底部地面互动光环 - 直径5m
      ============================================================ */}
      <group position={[0, 0.03, 0]}>
        {/* 1. 底板基座 */}
        <mesh ref={haloBaseRef} rotation={[-Math.PI / 2, 0, 0]}>
          <circleGeometry args={[2.55, 96]} />
          <meshStandardMaterial
            color="#0a1a23"
            transparent
            opacity={0.92}
            roughness={0.18}
            metalness={0.55}
            depthWrite={true}
          />
        </mesh>

        {/* 2. 三层主光环 - 青瓷 → 淡蓝绿 → 白 */}
        <mesh ref={haloRing1Ref} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
          <ringGeometry args={[2.36, 2.44, 128]} />
          <meshStandardMaterial
            color={COLORS.CELADON_GREEN}
            emissive={COLORS.CELADON_GREEN}
            emissiveIntensity={1.8}
            transparent
            opacity={0.72}
            side={THREE.DoubleSide}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
        <mesh ref={haloRing2Ref} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.012, 0]}>
          <ringGeometry args={[2.48, 2.52, 128]} />
          <meshStandardMaterial
            color={COLORS.LIGHT_BLUE_GREEN}
            emissive={COLORS.LIGHT_BLUE_GREEN}
            emissiveIntensity={1.4}
            transparent
            opacity={0.55}
            side={THREE.DoubleSide}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
          />
        </mesh>

        {/* 3. 内圈核心发光 */}
        <mesh position={[0, 0.016, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.48, 0.55, 64]} />
          <meshStandardMaterial
            color="#ffffff"
            emissive={COLORS.LIGHT_BLUE_GREEN}
            emissiveIntensity={1.1}
            transparent
            opacity={0.75}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
          />
        </mesh>

        {/* 4. 水墨水波扩散 - 10层同心圆周期错位 (减少5层,提升FPS) */}
        {[...Array(10)].map((_, i) => (
          <mesh
            key={`ripple-${i}`}
            ref={(el) => (rippleMeshRefs.current[i] = el)}
            rotation={[-Math.PI / 2, 0, 0]}
            position={[0, 0.022 + i * 0.0012, 0]}
            renderOrder={i + 10}
          >
            <ringGeometry args={[1.0, 1.03, 80]} />
            <meshBasicMaterial
              color={i % 2 === 0 ? COLORS.CELADON_GREEN : COLORS.LIGHT_BLUE_GREEN}
              transparent
              opacity={0.5}
              depthWrite={false}
              side={THREE.DoubleSide}
            />
          </mesh>
        ))}

        {/* 5. 钱塘潮涌动波 - S型流动光晕 */}
        <mesh ref={tideWaveRef} position={[0, 0.025, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[1.6, 1.76, 128]} />
          <meshBasicMaterial
            color={COLORS.LIGHT_BLUE_GREEN}
            transparent
            opacity={0.42}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
          />
        </mesh>

        {/* 6. 中心投影光斑 - 叠加强光 */}
        <mesh position={[0, 0.028, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <circleGeometry args={[1.4, 64]} />
          <meshBasicMaterial
            color="#ffffff"
            transparent
            opacity={0.16}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      </group>
    </group>
  )
}
