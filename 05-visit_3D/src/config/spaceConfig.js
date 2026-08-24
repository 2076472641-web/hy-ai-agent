// 浙江馆空间配置系统 - 21m × 13m × 6m
// 坐标系统: X轴(-10.5~10.5), Z轴(-6.5~6.5), Y轴(0~6)

export const SPACE = {
  // 展馆实际尺寸
  WIDTH: 21,       // X轴方向 - 长
  DEPTH: 13,       // Z轴方向 - 宽
  HEIGHT: 6,       // Y轴方向 - 高
  
  // 中心点在原点，方便计算
  HALF_WIDTH: 10.5,
  HALF_DEPTH: 6.5,
  
  // 主通道宽度
  MAIN_PATH_WIDTH: 3,
  SUB_PATH_WIDTH: 2,
}

// 区域划分 - 五大主题板块位置
export const ZONES = {
  // 中央装置
  CENTER: {
    x: 0, z: 0,
    radius: 2.5,  // 直径4-5m
  },
  
  // 源·文脉赓续 - 右前方 (X+, Z-)
  ORIGIN: {
    id: 'origin',
    name: '源·文脉赓续',
    subtitle: '浙江历史经典产业',
    position: { x: 6.5, z: -4 },
    size: { width: 6, depth: 3.5 },
    color: '#7fb3a5',  // 青瓷青
  },
  
  // 涌·科技筑能 - 左前方 (X-, Z-)
  SURGE: {
    id: 'surge',
    name: '涌·科技筑能',
    subtitle: '科技赋能·数字化·智能制造',
    position: { x: -6.5, z: -4 },
    size: { width: 6, depth: 3.5 },
    color: '#5a8cb5',  // 淡蓝
  },
  
  // 涟·聚链共生 - 左后方 (X-, Z+)
  RIPPLE: {
    id: 'ripple',
    name: '涟·聚链共生',
    subtitle: '产业链·产业集群·全球贸易',
    position: { x: -6.5, z: 4 },
    size: { width: 6, depth: 3.5 },
    color: '#6bb5a0',  // 青绿
  },
  
  // 汇·生态赋质 - 右后方 (X+, Z+)
  CONVERGE: {
    id: 'converge',
    name: '汇·生态赋质',
    subtitle: '政策·平台·人才·数字经济',
    position: { x: 6.5, z: 4 },
    size: { width: 6, depth: 3.5 },
    color: '#8a9bc4',  // 银蓝
  },
  
  // 润·境生万象 - 正后方 (Z+)
  NOURISH: {
    id: 'nourish',
    name: '润·境生万象',
    subtitle: '浙韵书房·经典出海·现代生活',
    position: { x: 0, z: 5.8 },
    size: { width: 10, depth: 2.5 },
    color: '#c4b89a',  // 暖米色
  },
}

// 入口位置 (21m长边中部 - Z最小边)
export const ENTRANCE = {
  position: { x: 0, y: 0, z: -6.5 },
  width: 5,
  height: 4.5,
}

// 主题配色系统
export const COLORS = {
  // 主色
  PORCELAIN_WHITE: '#f5f7f6',     // 瓷白
  CELADON_GREEN: '#7fb3a5',       // 青瓷青
  LIGHT_BLUE_GREEN: '#a8d0c8',    // 淡蓝绿
  
  // 辅助色
  SILVER_METAL: '#c8ccd0',        // 银色金属
  GLASS: 'rgba(200, 220, 230, 0.15)',  // 半透明玻璃
  WARM_LIGHT: '#ffd9a8',          // 暖色灯光
  
  // 背景
  DARK_BG: '#0a1520',
  FLOOR: '#1a2832',
}

// 镜头预设位置
export const CAMERA_PRESETS = {
  // 01 主入口人视
  ENTRANCE_VIEW: {
    name: '主入口人视',
    position: [0, 1.6, -6.8],
    target: [0, 1.8, -4],
  },
  // 02 入口45度
  ENTRANCE_ANGLE: {
    name: '入口45度',
    position: [6.5, 1.6, -5.8],
    target: [0, 1.6, -2.5],
  },
  // 03 中央装置
  CENTER_DEVICE: {
    name: '中央装置',
    position: [0, 2.8, 4.5],
    target: [0, 2.0, 0],
  },
  // 04 全馆鸟瞰
  BIRD_EYE: {
    name: '全馆鸟瞰',
    position: [0, 12, 0],
    target: [0, 0, 0],
  },
  // 05 展区视角
  EXHIBITION_VIEW: {
    name: '展区视角',
    position: [-6.5, 1.6, -1],
    target: [-6.5, 1.8, -4],
  },
  // 06 出口回望
  EXIT_LOOKBACK: {
    name: '出口回望',
    position: [0, 1.6, 6.8],
    target: [0, 1.8, -5],
  },
}

// 专业展示模式 - 6个建筑摄影镜头
// 1.7m人视高度, 55-60° FOV (24-26mm等效), 自然广角但无畸变
export const PRESENTATION_VIEWS = {
  // 01 入口正面 - 从入口外3m处正对入口门头, 展示门头+两侧立柱
  VIEW01_ENTRANCE_FRONT: {
    name: '01 入口正面',
    position: [0, 1.7, -10],
    target: [0, 3.0, -6.5],
    fov: 55,
  },
  // 02 入口45度 - 从入口侧前方45度, 看门头+侧展墙+入口内部
  VIEW02_ENTRANCE_45: {
    name: '02 入口45度',
    position: [8.5, 1.7, -8.5],
    target: [-1, 2.5, -5],
    fov: 55,
  },
  // 03 入口望中央 - 站在入口内侧, 看通道两侧展墙+远端装置
  VIEW03_ENTRANCE_TO_CORE: {
    name: '03 入口望中央',
    position: [0, 1.7, -5.5],
    target: [0, 2.5, 0],
    fov: 55,
  },
  // 04 中央装置英雄 - 装置英雄角度, 近距低角度仰视, 展现气势
  VIEW04_CORE_HERO: {
    name: '04 中央装置英雄',
    position: [2.5, 1.5, 2.5],
    target: [0, 2.8, 0],
    fov: 45,
  },
  // 05 全馆鸟瞰 - 9m高空俯视, 展示展馆整体布局
  VIEW05_BIRD_EYE: {
    name: '05 全馆鸟瞰',
    position: [0, 9, 0],
    target: [0, 0, 0],
    fov: 55,
  },
  // 06 展区回望 - 站在源展区内, 回望入口方向, 看到展柜+对侧展墙
  VIEW06_ZONE_LOOKBACK: {
    name: '06 展区回望',
    position: [7.5, 1.7, -3.5],
    target: [0, 2.2, -5.5],
    fov: 55,
  },
}

// 源·文脉赓续 - 历史经典产业数据
export const HERITAGE_INDUSTRIES = [
  { name: '茶叶', region: '杭州/绍兴', icon: 'tea' },
  { name: '丝绸', region: '杭州/湖州', icon: 'silk' },
  { name: '黄酒', region: '绍兴', icon: 'wine' },
  { name: '青瓷', region: '龙泉/越州', icon: 'porcelain' },
  { name: '宝剑', region: '龙泉', icon: 'sword' },
  { name: '中药', region: '杭州/金华', icon: 'herb' },
  { name: '火腿', region: '金华', icon: 'ham' },
  { name: '木雕', region: '东阳', icon: 'wood' },
  { name: '石雕', region: '青田', icon: 'stone' },
  { name: '根雕', region: '开化', icon: 'root' },
  { name: '文房', region: '湖州/歙县', icon: 'stationery' },
  { name: '珍珠', region: '诸暨', icon: 'pearl' },
  { name: '金属雕', region: '温州/绍兴', icon: 'metal' },
]
