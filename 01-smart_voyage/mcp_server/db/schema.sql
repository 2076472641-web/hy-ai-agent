# ============================================================
# 途景汇文旅多智能体系统 - MySQL 数据库 Schema
# ============================================================

CREATE DATABASE IF NOT EXISTS songjing_hui
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE songjing_hui;

# -----------------------------------------------------------
# 1. 目的地城市表
# -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS destinations (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(50)  NOT NULL COMMENT '城市名称',
    level       VARCHAR(20)  DEFAULT '二线' COMMENT '城市等级：一线/新一线/二线',
    province    VARCHAR(50)  DEFAULT '' COMMENT '所属省份',
    description TEXT COMMENT '城市简介',
    best_season VARCHAR(100) DEFAULT '' COMMENT '最佳游玩季节',
    tags        JSON         COMMENT '标签数组，如["历史文化","美食"]',
    created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_name (name)
) ENGINE=InnoDB COMMENT='目的地城市表';

# -----------------------------------------------------------
# 2. 景点表
# -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS attractions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(100) NOT NULL COMMENT '景点名称',
    city_id         INT          NOT NULL COMMENT '所属城市ID',
    category        VARCHAR(50)  DEFAULT '' COMMENT '分类：自然风光/历史文化/主题乐园/博物馆/美食',
    rating          DECIMAL(2,1) DEFAULT 0.0 COMMENT '评分 0-5',
    ticket_price    VARCHAR(50)  DEFAULT '免费' COMMENT '门票价格',
    duration        VARCHAR(50)  DEFAULT '' COMMENT '建议游玩时长',
    description     TEXT COMMENT '景点介绍',
    opening_hours   VARCHAR(200) DEFAULT '' COMMENT '开放时间',
    address         VARCHAR(300) DEFAULT '' COMMENT '地址',
    latitude        DECIMAL(10,7) DEFAULT NULL COMMENT '纬度',
    longitude       DECIMAL(10,7) DEFAULT NULL COMMENT '经度',
    tips            JSON         COMMENT '游玩小贴士',
    images          JSON         COMMENT '图片URL列表',
    status          TINYINT     DEFAULT 1 COMMENT '状态：1-启用 0-禁用',
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_city (city_id),
    INDEX idx_category (category),
    INDEX idx_rating (rating),
    FULLTEXT KEY ft_name_desc (name, description)
) ENGINE=InnoDB COMMENT='景点信息表';

# -----------------------------------------------------------
# 3. 交通线路表
# -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS transport_routes (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    origin_city_id  INT          NOT NULL COMMENT '出发城市ID',
    dest_city_id    INT          NOT NULL COMMENT '目的城市ID',
    transport_type  VARCHAR(20)  NOT NULL COMMENT '交通方式：飞机/高铁/汽车/地铁',
    duration_min    INT          DEFAULT 0 COMMENT '耗时（分钟）',
    price_standard   DECIMAL(8,2) DEFAULT 0 COMMENT '标准价',
    price_discount   DECIMAL(8,2) DEFAULT 0 COMMENT '折扣价',
    frequency       VARCHAR(100) DEFAULT '' COMMENT '班次频率',
    departure       VARCHAR(200) DEFAULT '' COMMENT '出发站/机场',
    arrival         VARCHAR(200) DEFAULT '' COMMENT '到达站/机场',
    tips            TEXT COMMENT '乘坐小贴士',
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_origin_dest (origin_city_id, dest_city_id),
    INDEX idx_type (transport_type)
) ENGINE=InnoDB COMMENT='交通线路表';

# -----------------------------------------------------------
# 4. 旅游套餐表
# -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS travel_packages (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(200) NOT NULL COMMENT '套餐名称',
    destination     VARCHAR(100) NOT NULL COMMENT '目的地',
    days            INT          NOT NULL COMMENT '天数',
    nights          INT          DEFAULT 0 COMMENT '夜数',
    price_economy   DECIMAL(10,2) DEFAULT 0 COMMENT '经济版价格',
    price_standard  DECIMAL(10,2) DEFAULT 0 COMMENT '标准版价格',
    price_luxury    DECIMAL(10,2) DEFAULT 0 COMMENT '豪华版价格',
    highlights      JSON         COMMENT '亮点数组',
    itinerary       JSON         COMMENT '行程详情',
    included        JSON         COMMENT '费用包含',
    excluded        JSON         COMMENT '费用不含',
    status          TINYINT     DEFAULT 1 COMMENT '状态：1-上架 0-下架',
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_destination (destination),
    INDEX idx_days (days)
) ENGINE=InnoDB COMMENT='旅游套餐表';

# -----------------------------------------------------------
# 5. 对话历史表（用于持久化对话记录）
# -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversation_history (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id      VARCHAR(100) NOT NULL COMMENT '会话ID',
    user_id         VARCHAR(100) DEFAULT '' COMMENT '用户ID',
    role            VARCHAR(20)  NOT NULL COMMENT '消息角色：user/assistant/system',
    content         TEXT COMMENT '消息内容',
    tool_calls      JSON         COMMENT '工具调用信息',
    tool_results    JSON         COMMENT '工具调用结果',
    tokens_used     INT          DEFAULT 0 COMMENT 'Token使用量',
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_user (user_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB COMMENT='对话历史表';

# -----------------------------------------------------------
# 6. 智能体任务记录表
# -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_tasks (
    id              VARCHAR(100) PRIMARY KEY COMMENT '任务ID',
    task_type       VARCHAR(50)  NOT NULL COMMENT '任务类型',
    title           VARCHAR(500) DEFAULT '' COMMENT '任务标题',
    status          VARCHAR(20)  DEFAULT 'pending' COMMENT '状态：pending/in_progress/completed/failed',
    assigned_to     VARCHAR(50)  DEFAULT '' COMMENT '指派的智能体',
    parent_task_id  VARCHAR(100) DEFAULT NULL COMMENT '父任务ID',
    result          JSON         COMMENT '任务结果',
    error_message   TEXT COMMENT '错误信息',
    metadata        JSON         COMMENT '元数据',
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_assigned (assigned_to)
) ENGINE=InnoDB COMMENT='智能体任务记录表';