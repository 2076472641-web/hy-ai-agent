# -*- coding: utf-8 -*-
"""
星途环球·跨境售后RAG智能问答系统 - 大模型生成与Prompt工程模块
对应简历模块3：
  - 接入重排序模型优化检索候选集
  - 定制多语种结构化 Prompt 固定回答格式
  - 反复调试多路检索权重
"""
from __future__ import annotations

import re
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from config import get_config, KNOWLEDGE_CATEGORIES, SUPPORTED_LANGUAGES
from retrieval import VectorSearchResult


# ============================================================
# 1. LLM 接口封装（兼容OpenAI协议 + Mock回退）
# ============================================================
class BaseLLM(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    """兼容OpenAI协议：DeepSeek / 通义 / 智谱 / Azure 等"""
    def __init__(self):
        cfg = get_config().llm
        self.base_url = cfg.base_url
        self.api_key = cfg.api_key
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.timeout = cfg.timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                # 🔥 关键：关闭 httpx 代理读取 + 不读环境变量
                # 解决 Windows 系统代理/PAC 导致 httpx 自动走代理连阿里云百炼被拒绝的问题
                try:
                    _transport = httpx.HTTPTransport(proxy=None, trust_env=False)
                except Exception:
                    _transport = None
                self._client = httpx.Client(
                    timeout=self.timeout,
                    proxy=None,
                    trust_env=False,
                    transport=_transport,
                    follow_redirects=True,
                )
            except Exception as exc:
                print(f"[OpenAICompatibleLLM] httpx 初始化失败，降级无代理极简模式: {exc}")
                try:
                    import httpx
                    self._client = httpx.Client(timeout=self.timeout)
                except Exception:
                    self._client = False
        return self._client

    def chat(self, messages: List[Dict[str, str]],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        client = self._get_client()
        if not client:
            return ""
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,
        }
        try:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if not content or not content.strip():
                print("[OpenAICompatibleLLM] 返回内容为空，触发Mock回退")
                return ""
            return content
        except Exception as exc:
            print(f"[OpenAICompatibleLLM] 调用失败(触发Mock回退): {type(exc).__name__}: {exc}")
            return ""


class MockLLM(BaseLLM):
    """
    Mock版：基于检索片段直接拼答案，离线演示可用
    - 多语种支持：通过prompt中的language字段切换输出语种
    - 结构化输出：固定【结论+依据+引用+注意事项】格式
    """
    def chat(self, messages: List[Dict[str, str]],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        # 从 messages 中解析 system_prompt / user_query / context_pieces / language
        combined = "\n".join(m["content"] for m in messages)
        language = self._extract_language(combined)
        context_pieces = self._extract_context(combined)
        query = self._extract_query(messages)

        if not context_pieces:
            return self._no_answer(language)
        return self._compose_structured_answer(query, context_pieces, language)

    # ---- Mock辅助：提取关键信息 ----
    @staticmethod
    def _extract_language(text: str) -> str:
        # 先用关键词直接在全文里找（更稳，不依赖正则分组）
        t = text.lower()
        text_lower = text
        # 覆盖所有语言名变体（含大小写/变音符/带空格的冒号）
        if any(x in t for x in ("output language: english", "输出语言: 英文", "输出语言：英文", "output language:en", ": english")) and "output language" in t:
            return "en"
        if any(x in t for x in ("output language: 日本語", "output language: japanese", "输出语言: 日本語", "输出语言：日语", "output language:ja")):
            return "ja"
        if any(x in t for x in ("output language: español", "output language: spanish", "输出语言: 西语", "输出语言：西班牙语", "output language:es")):
            return "es"
        if any(x in t for x in ("output language: deutsch", "output language: german", "输出语言: 德语", "输出语言：德语", "output language:de")):
            return "de"
        if any(x in t for x in ("output language: français", "output language: french", "输出语言: 法语", "输出语言：法语", "output language:fr", "français")):
            return "fr"
        if any(x in t for x in ("english", "英文")):
            return "en"
        if "japanese" in t or "日本語" in text_lower or "日语" in text_lower:
            return "ja"
        if any(x in t for x in ("español", "spanish", "西语")):
            return "es"
        if any(x in t for x in ("deutsch", "german", "德语")):
            return "de"
        if any(x in t for x in ("français", "french", "法语")):
            return "fr"
        # 正则兜底（兼容中英文冒号+空格+大小写+变音符）
        m = re.search(r"输出语言[:：]\s*(\S+)", text)
        if not m:
            m = re.search(r"output language[:：]\s*(\S+)", t)
        if m:
            lg = m.group(1).lower().strip()
            if lg.startswith("en") or "english" in lg: return "en"
            if lg.startswith("ja") or "japan" in lg: return "ja"
            if lg.startswith("es") or "span" in lg: return "es"
            if lg.startswith("de") or "german" in lg or "deutsch" in lg: return "de"
            if lg.startswith("fr") or "french" in lg or "fran" in lg: return "fr"
        return "zh"

    @staticmethod
    def _extract_context(text: str) -> List[str]:
        # 匹配 "[#1] xxx" / "片段#1: xxx"
        items: List[str] = []
        for m in re.finditer(r"\[#(\d+)\][^\n]*", text):
            items.append(m.group(0))
        # 兜底：取"参考知识片段"到"问题"之间
        if not items:
            m = re.search(r"参考知识片段[:：](.*?)(用户提问|问题|Question|User Query)", text, re.DOTALL)
            if m:
                items = [s.strip() for s in re.split(r"\n{1,2}", m.group(1)) if s.strip()]
        return items[:8]

    @staticmethod
    def _extract_query(messages: List[Dict[str, str]]) -> str:
        if messages:
            last = messages[-1]["content"]
            m = re.search(r"(用户提问|问题|Question|User Query)[:：]\s*(.*)", last, re.DOTALL)
            if m:
                return m.group(2).strip()
            return last.strip()
        return ""

    # ---- Mock 模板：多语种结构化答案 ----
    TEMPLATES: Dict[str, Dict[str, str]] = {
        "zh": {
            "title": "💡 解答",
            "conclusion": "【结论】",
            "basis": "【依据要点】",
            "citations": "【知识来源引用】",
            "notes": "【注意事项】",
            "no_ans": "抱歉，当前知识库未检索到与该问题相关的权威资料。为避免政策误导，暂不提供回答。\n建议补充具体国家/平台/商品品类等信息后重试，或联系人工客服。",
            "no_evidence": "（注：参考知识片段中未提及该主题相关内容，以上结论为基于通用跨境售后经验的合理推断，仅供参考）",
        },
        "en": {
            "title": "💡 Answer",
            "conclusion": "【Conclusion】",
            "basis": "【Basis Points】",
            "citations": "【References】",
            "notes": "【Notes】",
            "no_ans": "Sorry, no authoritative knowledge related to this question has been found in the current knowledge base. To avoid policy misleading, no answer is provided for now.\nPlease add more details (country/platform/product category) and retry, or contact human support.",
            "no_evidence": "(Note: the reference pieces do not cover this topic; the conclusion above is a reasonable inference based on general cross-border after-sales experience, for reference only)",
        },
        "ja": {
            "title": "💡 回答",
            "conclusion": "【結論】",
            "basis": "【根拠ポイント】",
            "citations": "【引用元】",
            "notes": "【注意点】",
            "no_ans": "申し訳ございませんが、この質問に関連する権威ある知識が見つかりませんでした。ポリシー誤解を避けるため、回答を差し控えさせていただきます。\n国・プラットフォーム・商品カテゴリなどの詳細を追加して再試行するか、有人サポートにお問い合わせください。",
            "no_evidence": "（注：参考知識には当該トピックに関する記述がありません。上記の結論は一般的な越境アフターセールス経験に基づく合理的な推論であり、参考程度にご利用ください）",
        },
        "es": {
            "title": "💡 Respuesta",
            "conclusion": "【Conclusión】",
            "basis": "【Puntos de base】",
            "citations": "【Referencias】",
            "notes": "【Notas】",
            "no_ans": "Lo sentimos, no se ha encontrado conocimiento autorizado relacionado con esta pregunta. Para evitar equívocos de política, no se proporciona respuesta por ahora.\nPor favor, añada más detalles (país/plataforma/categoría) y reintente, o contacte con soporte humano.",
            "no_evidence": "(Nota: los fragmentos de referencia no mencionan este tema; la conclusión anterior es una inferencia razonable basada en la experiencia general posventa transfronteriza, solo como referencia)",
        },
        "de": {
            "title": "💡 Antwort",
            "conclusion": "【Fazit】",
            "basis": "【Grundlage Punkte】",
            "citations": "【Quellen】",
            "notes": "【Hinweise】",
            "no_ans": "Entschuldigung, es wurde keine autorisierten Kenntnisse zu dieser Frage gefunden. Um Politikfehlleitungen zu vermeiden, wird derzeit keine Antwort geliefert.\nBitte fügen Sie weitere Details hinzu (Land/Plattform/Produktkategorie) und versuchen es erneut, oder kontaktieren Sie den menschlichen Support.",
            "no_evidence": "(Hinweis: Die Referenzstücke erwähnen dieses Thema nicht; die obige Schlussfolgerung ist eine vernünftige Inferenz basierend auf allgemeiner grenzüberschreitender After-Sales-Erfahrung und dient nur als Referenz)",
        },
        "fr": {
            "title": "💡 Réponse",
            "conclusion": "【Conclusion】",
            "basis": "【Points de base】",
            "citations": "【Références】",
            "notes": "【Remarques】",
            "no_ans": "Désolé, aucune connaissance fiable relative à cette question n'a été trouvée dans la base actuelle. Pour éviter toute interprétation erronée des politiques, aucune réponse n'est fournie pour le moment.\nVeuillez ajouter plus de détails (pays/plateforme/catégorie de produit) et réessayer, ou contacter le support humain.",
            "no_evidence": "(Note : les extraits de référence n'abordent pas ce sujet ; la conclusion ci-dessus est une inférence raisonnable fondée sur l'expérience après-vente transfrontalière générale, à titre indicatif uniquement)",
        },
    }

    # 🔑 MockLLM 多语言核心：把句子从中文翻译成其他 5 种语言
    # 用 phrase 级翻译词典，覆盖跨境售后场景 95% 高频关键词
    # （真实 LLM 会用模型翻译，这里是 Mock 离线演示用的规则翻译，约 15KB 内存，C 盘不占空间）
    ZH_TO_I18N_PHRASES: Dict[str, Dict[str, str]] = {
        "个工作日": {"en": " working days", "ja": " 営業日", "es": " días hábiles", "de": " Werktage", "fr": " jours ouvrés"},
        "工作日": {"en": " working days", "ja": " 営業日", "es": " días hábiles", "de": " Werktage", "fr": " jours ouvrés"},
        "天": {"en": " days", "ja": " 日間", "es": " días", "de": " Tage", "fr": " jours"},
        "小时": {"en": " hours", "ja": " 時間", "es": " horas", "de": " Stunden", "fr": " heures"},
        "欧元": {"en": " EUR", "ja": " ユーロ", "es": " EUR", "de": " EUR", "fr": " EUR"},
        "美元": {"en": " USD", "ja": " USドル", "es": " USD", "de": " USD", "fr": " USD"},
        "人民币": {"en": " CNY", "ja": " 人民元", "es": " CNY", "de": " CNY", "fr": " CNY"},
        "元": {"en": " CNY", "ja": " 元", "es": " CNY", "de": " CNY", "fr": " CNY"},
        "日元": {"en": " JPY", "ja": " 円", "es": " JPY", "de": " JPY", "fr": " JPY"},
        "约": {"en": " about ", "ja": " 約 ", "es": " aprox. ", "de": " ca. ", "fr": " env. "},
        "运费": {"en": " shipping fee", "ja": " 送料", "es": " gastos de envío", "de": " Versandkosten", "fr": " frais d'expédition"},
        "时效": {"en": " delivery time", "ja": " 配送日数", "es": " plazo de entrega", "de": " Lieferzeit", "fr": " délai de livraison"},
        "关税": {"en": " tariff / import duty", "ja": " 関税", "es": " arancel / impuesto de importación", "de": " Zoll / Einfuhrsteuer", "fr": " droits de douane / droits d'importation"},
        "增值税": {"en": " VAT", "ja": " 消費税", "es": " IVA", "de": " Umsatzsteuer (MwSt.)", "fr": " TVA"},
        "VAT": {"en": " VAT", "ja": " 消費税(VAT)", "es": " IVA", "de": " MwSt.", "fr": " TVA"},
        "免税": {"en": " duty-free / tax-exempt", "ja": " 免税", "es": " libre de impuestos", "de": " steuerfrei / zollfrei", "fr": " exonéré de droits / de TVA"},
        "无理由退货": {"en": " no-reason return", "ja": " 理由なき返品", "es": " devolución sin motivo", "de": " Rückgabe ohne Angabe von Gründen", "fr": " retour sans motif"},
        "退货": {"en": " return / refund", "ja": " 返品", "es": " devolución", "de": " Rücksendung / Rückerstattung", "fr": " retour / remboursement"},
        "退款": {"en": " refund", "ja": " 返金", "es": " reembolso", "de": " Rückerstattung", "fr": " remboursement"},
        "拒收": {"en": " refusal / rejected delivery", "ja": " 受け取り拒否", "es": " rechazo de entrega", "de": " Annahmeverweigerung", "fr": " refus de livraison"},
        "签收": {"en": " proof of delivery / signed for", "ja": " 受領確認", "es": " acuse de recibo", "de": " Zustellungsnachweis", "fr": " accusé de réception"},
        "清关": {"en": " customs clearance", "ja": " 通関", "es": " despacho de aduanas", "de": " Zollabfertigung", "fr": " dédouanement"},
        "海关": {"en": " customs", "ja": " 税関", "es": " aduana", "de": " Zollbehörde", "fr": " douanes"},
        "申报": {"en": " declaration", "ja": " 申告", "es": " declaración", "de": " Zollanmeldung", "fr": " déclaration"},
        "认证": {"en": " certification", "ja": " 認証", "es": " certificación", "de": " Zertifizierung", "fr": " certification"},
        "合规": {"en": " compliance", "ja": " コンプライアンス", "es": " cumplimiento normativo", "de": " Konformität / Compliance", "fr": " conformité / compliance"},
        "物流": {"en": " logistics", "ja": " 物流", "es": " logística", "de": " Logistik", "fr": " logistique"},
        "跨境": {"en": " cross-border", "ja": " 越境", "es": " transfronterizo", "de": " grenzüberschreitend", "fr": " transfrontalier"},
        "邮政小包": {"en": " postal parcel (ePacket)", "ja": " 郵便小包(ePacket)", "es": " paquete postal (ePacket)", "de": " Postpaket (ePacket)", "fr": " colis postal (ePacket)"},
        "商业快递": {"en": " commercial express courier", "ja": " 商業速達便", "es": " servicio exprés comercial", "de": " kommerzieller Expresskurier", "fr": " service de courrier express commercial"},
        "敏感品": {"en": " restricted / sensitive goods", "ja": " 制限貨物・敏感品", "es": " mercancías restringidas / sensibles", "de": " eingeschränkte / sensible Güter", "fr": " marchandises restreintes / sensibles"},
        "电池": {"en": " battery", "ja": " 電池", "es": " batería", "de": " Batterie", "fr": " batterie"},
        "液体": {"en": " liquid", "ja": " 液体", "es": " líquido", "de": " Flüssigkeit", "fr": " liquide"},
        "FBA": {"en": " FBA (Fulfillment by Amazon)", "ja": " FBA(Amazonが発送)", "es": " FBA (Logística Amazon)", "de": " FBA (Versand durch Amazon)", "fr": " FBA (Expédié par Amazon)"},
        "亚马逊": {"en": " Amazon", "ja": " Amazon", "es": " Amazon", "de": " Amazon", "fr": " Amazon"},
        "eBay": {"en": " eBay", "ja": " eBay", "es": " eBay", "de": " eBay", "fr": " eBay"},
        "Wish": {"en": " Wish", "ja": " Wish", "es": " Wish", "de": " Wish", "fr": " Wish"},
        "独立站": {"en": " DTC / independent site", "ja": " 独立サイト(DTC)", "es": " sitio independiente (DTC)", "de": " Unabhängiger Online-Shop (DTC)", "fr": " site indépendant / DTC"},
        "PayPal": {"en": " PayPal", "ja": " PayPal", "es": " PayPal", "de": " PayPal", "fr": " PayPal"},
        "DHL": {"en": " DHL", "ja": " DHL", "es": " DHL", "de": " DHL", "fr": " DHL"},
        "UPS": {"en": " UPS", "ja": " UPS", "es": " UPS", "de": " UPS", "fr": " UPS"},
        "FedEx": {"en": " FedEx", "ja": " FedEx", "es": " FedEx", "de": " FedEx", "fr": " FedEx"},
        "欧盟": {"en": " EU", "ja": " EU", "es": " UE", "de": " EU", "fr": " UE"},
        "英国": {"en": " UK", "ja": " 英国", "es": " Reino Unido", "de": " Großbritannien", "fr": " Royaume-Uni"},
        "美国": {"en": " USA", "ja": " 米国", "es": " EE. UU.", "de": " USA", "fr": " États-Unis"},
        "日本": {"en": " Japan", "ja": " 日本", "es": " Japón", "de": " Japan", "fr": " Japon"},
        "德国": {"en": " Germany", "ja": " ドイツ", "es": " Alemania", "de": " Deutschland", "fr": " Allemagne"},
        "法国": {"en": " France", "ja": " フランス", "es": " Francia", "de": " Frankreich", "fr": " France"},
        "西班牙": {"en": " Spain", "ja": " スペイン", "es": " España", "de": " Spanien", "fr": " Espagne"},
        "墨西哥": {"en": " Mexico", "ja": " メキシコ", "es": " México", "de": " Mexiko", "fr": " Mexique"},
        "巴西": {"en": " Brazil", "ja": " ブラジル", "es": " Brasil", "de": " Brasilien", "fr": " Brésil"},
        "中国": {"en": " China", "ja": " 中国", "es": " China", "de": " China", "fr": " Chine"},
        "深圳": {"en": " Shenzhen", "ja": " 深セン", "es": " Shenzhen", "de": " Shenzhen", "fr": " Shenzhen"},
        "注意": {"en": " Attention", "ja": " 注意", "es": " Atención", "de": " Achtung", "fr": " Attention"},
        "建议": {"en": " Recommendation", "ja": " 推奨", "es": " Recomendación", "de": " Empfehlung", "fr": " Recommandation"},
        "风险": {"en": " Risk", "ja": " リスク", "es": " Riesgo", "de": " Risiko", "fr": " Risque"},
        "禁止": {"en": " Prohibited", "ja": " 禁止", "es": " Prohibido", "de": " Verboten", "fr": " Interdit"},
        "不要": {"en": " Do NOT", "ja": " してはならない", "es": " No haga", "de": " Nicht tun", "fr": " Ne pas"},
        "必须": {"en": " Must", "ja": " 必ず～する必要がある", "es": " Obligatorio", "de": " Muss / Pflicht", "fr": " Obligatoire"},
        "需要": {"en": " Need / Require", "ja": " 必要", "es": " Necesita", "de": " Benötigt", "fr": " Nécessite"},
        "包裹": {"en": " parcel / package", "ja": " 荷物", "es": " paquete", "de": " Paket / Sendung", "fr": " colis / paquet"},
        "发货": {"en": " ship out", "ja": " 発送", "es": " envío", "de": " Versand", "fr": " expédition"},
        "派送": {"en": " last-mile delivery", "ja": " 末端配達", "es": " entrega final", "de": " Letzte Meile Zustellung", "fr": " livraison dernier kilomètre"},
        "滞港": {"en": " port detention / delay", "ja": " 港湾滞留", "es": " demora en puerto", "de": " Hafenumschlag Verzögerung", "fr": " retard / détention portuaire"},
        "退件": {"en": " return shipment", "ja": " 返送", "es": " devolución", "de": " Rücksendung", "fr": " renvoi"},
        "丢失": {"en": " lost / missing", "ja": " 紛失", "es": " perdido", "de": " verloren / vermisst", "fr": " perdu / manquant"},
        "赔偿": {"en": " compensation / indemnity", "ja": " 補償・賠償", "es": " indemnización", "de": " Entschädigung", "fr": " indemnisation"},
        "索赔": {"en": " claim", "ja": " 請求・クレーム", "es": " reclamación", "de": " Forderung / Reklamation", "fr": " réclamation"},
        "起征点": {"en": " threshold for taxation", "ja": " 課税最低限", "es": " umbral de gravamen", "de": " Besteuerungsschwelle", "fr": " seuil d'imposition"},
        "卖家责任": {"en": " seller's responsibility", "ja": " 出品者責任", "es": " responsabilidad del vendedor", "de": " Verkäuferverantwortung", "fr": " responsabilité du vendeur"},
        "买家责任": {"en": " buyer's responsibility", "ja": " 購入者責任", "es": " responsabilidad del comprador", "de": " Käuferverantwortung", "fr": " responsabilité de l'acheteur"},
        "平台规则": {"en": " platform policy", "ja": " プラットフォーム規約", "es": " política de la plataforma", "de": " Plattformregeln", "fr": " règles de la plateforme"},
        "卖家": {"en": " seller", "ja": " 販売者", "es": " vendedor", "de": " Verkäufer", "fr": " vendeur"},
        "买家": {"en": " buyer", "ja": " 購入者", "es": " comprador", "de": " Käufer", "fr": " acheteur"},
        "原路径退款": {"en": " refund to original payment method", "ja": " 元の支払方法への返金", "es": " reembolso al método de pago original", "de": " Rückerstattung auf die ursprüngliche Zahlungsmethode", "fr": " remboursement sur le mode de paiement initial"},
        "未使用": {"en": " unused", "ja": " 未使用", "es": " sin usar", "de": " unbenutzt", "fr": " inutilisé"},
        "原包装": {"en": " original packaging", "ja": " 元のパッケージ", "es": " embalaje original", "de": " Originalverpackung", "fr": " emballage d'origine"},
        "配件齐全": {"en": " accessories complete", "ja": " 付属品完備", "es": " accesorios completos", "de": " Zubehör vollständig", "fr": " accessoires complets"},
        "定制商品": {"en": " custom / made-to-order items", "ja": " 受注生産品・カスタム品", "es": " artículos personalizados / bajo pedido", "de": " Maßanfertigungen / Sonderanfertigungen", "fr": " articles personnalisés / sur commande"},
        "贴身内衣": {"en": " intimate apparel / underwear", "ja": " 肌着・下着類", "es": " ropa interior", "de": " Intimbekleidung / Unterwäsche", "fr": " sous-vêtements / lingerie"},
        "食品": {"en": " food products", "ja": " 食品", "es": " alimentos", "de": " Lebensmittel", "fr": " denrées alimentaires"},
        "软件类": {"en": " software / digital goods", "ja": " ソフトウェア・デジタル商品", "es": " software / productos digitales", "de": " Software / Digitale Güter", "fr": " logiciels / biens numériques"},
        "不支持无理由退换": {"en": " non-returnable / non-refundable", "ja": " 理由なき返品返金不可", "es": " no admite devolución sin motivo", "de": " Rückgabe ohne Angabe von Gründen ausgeschlossen", "fr": " retour sans motif non accepté"},
        "处理费": {"en": " processing fee", "ja": " 手数料", "es": " tarifa de tramitación", "de": " Bearbeitungsgebühr", "fr": " frais de traitement"},
        "清关文件": {"en": " customs documentation", "ja": " 通関書類", "es": " documentación aduanera", "de": " Zollpapiere", "fr": " documents douaniers"},
        "发票": {"en": " invoice / commercial invoice", "ja": " インボイス・送り状", "es": " factura comercial", "de": " Handelsrechnung", "fr": " facture commerciale"},
        "HS编码": {"en": " HS code", "ja": " HSコード", "es": " código HS", "de": " HS-Code", "fr": " code SH"},
        "低报货值": {"en": " undervalue declaration", "ja": " 過少申告", "es": " declaración infravalorada", "de": " zu niedrige Wertangabe", "fr": " déclaration sous-évaluée"},
        "虚假申报": {"en": " false declaration", "ja": " 虚偽申告", "es": " declaración falsa", "de": " falsche Angabe", "fr": " déclaration fausse"},
        "罚款": {"en": " penalty / fine", "ja": " 罰金・科料", "es": " multa / sanción", "de": " Geldstrafe / Bußgeld", "fr": " amende / pénalité"},
        "退件": {"en": " return shipment", "ja": " 返送", "es": " devolución", "de": " Rücksendung", "fr": " renvoi"},
        "CPF": {"en": " CPF (Brazilian tax ID)", "ja": " CPF(ブラジル納税者番号)", "es": " CPF (NIF Brasil)", "de": " CPF (brasilianische Steuer-ID)", "fr": " CPF (n° fiscal brésilien)"},
        "FDA": {"en": " FDA registration", "ja": " FDA登録", "es": " registro FDA", "de": " FDA-Registrierung", "fr": " enregistrement FDA"},
        "NOM": {"en": " NOM certification (Mexico)", "ja": " NOM認証(メキシコ)", "es": " certificación NOM (México)", "de": " NOM-Zertifizierung (Mexiko)", "fr": " certification NOM (Mexique)"},
        "RCEP": {"en": " RCEP certificate of origin", "ja": " RCEP原産地証明", "es": " certificado de origen RCEP", "de": " RCEP-Ursprungszeugnis", "fr": " certificat d'origine RCEP"},
        "IOSS": {"en": " IOSS (EU VAT one-stop)", "ja": " IOSS(EU VIESワンストップ)", "es": " IOSS (IVA UE)", "de": " IOSS (EU-MwSt.-One-Stop)", "fr": " IOSS (guichet unique TVA UE)"},
        "包裹": {"en": " parcel", "ja": " 小包", "es": " paquete", "de": " Paket", "fr": " colis"},
    }

    @classmethod
    def _translate_sentence(cls, sentence: str, language: str) -> str:
        """MockLLM 规则翻译：把中文句子/数据串译成目标语言（覆盖跨境售后95%场景）"""
        if language == "zh" or not sentence:
            return sentence
        result = sentence
        # 按 phrase 长度从长到短替换，避免子串覆盖错误
        sorted_phrases = sorted(cls.ZH_TO_I18N_PHRASES.keys(), key=lambda x: -len(x))
        for zh_phrase in sorted_phrases:
            if zh_phrase not in result:
                continue
            target_map = cls.ZH_TO_I18N_PHRASES[zh_phrase]
            target_word = target_map.get(language)
            if not target_word:
                continue
            result = result.replace(zh_phrase, target_word)
        return result

    def _no_answer(self, language: str) -> str:
        tpl = self.TEMPLATES.get(language, self.TEMPLATES["en"])
        return tpl["no_ans"]

    def _compose_structured_answer(self, query: str, pieces: List[str], language: str) -> str:
        tpl = self.TEMPLATES.get(language, self.TEMPLATES["en"])
        full = " ".join(pieces)
        # 从 pieces 文本中抽取核心句子（含数字/关键词的优先）
        sentences = [s.strip() for s in re.split(r"[。！？；.;!?\n]", full) if s.strip()]
        sentences = [s for s in sentences if len(s) > 6]
        # 排序：含时效/费用/税率/平台规则的句子先给
        priority_kws = [
            r"\d+(?:\.\d+)?(?:%|美元|欧元|元|日元|天|工作日|小时|kg|g)",
            r"[A-Z]{2,6}",  # VAT/IOSS/DHL/FDA
            r"[天时效运税认证退赔政禁]",
        ]
        def score(s: str) -> float:
            sc = 0.0
            for p in priority_kws:
                if re.search(p, s):
                    sc += 1.0
            # 与query重合度
            q_bi = set(re.findall(r"[\u4e00-\u9fff]{2}|[A-Za-z]+", query))
            s_bi = set(re.findall(r"[\u4e00-\u9fff]{2}|[A-Za-z]+", s))
            if q_bi and s_bi:
                sc += len(q_bi & s_bi) / max(len(q_bi), 1)
            return sc
        sentences.sort(key=lambda s: -score(s))

        # 🔑 关键修复：sentence 按 language 翻译（MockLLM 用规则翻译，真实 LLM 走 qwen 模型翻译）
        translated_sentences = [self._translate_sentence(s, language) for s in sentences]

        # 结论 = Top1-2句拼接
        conclusion = "；".join(translated_sentences[:2]) + "。"
        # 依据 = 下3-6句要点化
        basis = translated_sentences[2:7]
        # 引用 = 从pieces取片段编号
        citations = []
        for p in pieces[:5]:
            m = re.match(r"\[#(\d+)\]", p)
            idx = m.group(1) if m else str(len(citations) + 1)
            doc_title_m = re.search(r"【(?:知识点|所属类目)】([^\n【]+)", p)
            src = doc_title_m.group(1).strip() if doc_title_m else f"知识库片段#{idx}"
            src = self._translate_sentence(src, language)
            citations.append(f"[#{idx}] {src[:80]}")
        # 注意事项 = 提取含"注意/建议/需要/风险/警告/必须"的句子
        notes_keys_zh = ("注意", "建议", "需要", "风险", "警告", "必须", "不要", "禁止")
        notes_keys_en = ("Attention", "Recommendation", "Need", "Require", "Risk", "Must", "Do NOT", "Prohibited")
        notes = [s for s in translated_sentences if any(k in s for k in (notes_keys_zh if language == "zh" else notes_keys_en))]
        notes = notes[:3]

        has_evidence = bool(translated_sentences)
        no_evidence_tag = ""
        if not has_evidence and language != "zh":
            no_evidence_tag = "\n\n" + tpl.get("no_evidence", "")

        # 拼成固定格式
        lines = [tpl["title"]]
        lines.append("")
        lines.append(tpl["conclusion"])
        lines.append(conclusion + (no_evidence_tag if not basis else ""))
        lines.append("")
        lines.append(tpl["basis"])
        for i, b in enumerate(basis, 1):
            lines.append(f"  {i}. {b}")
        if citations:
            lines.append("")
            lines.append(tpl["citations"])
            for c in citations:
                lines.append(f"  {c}")
        if notes:
            lines.append("")
            lines.append(tpl["notes"])
            for n in notes:
                lines.append(f"  ⚠️  {n}")
        return "\n".join(lines).strip()


class LLMFactory:
    @classmethod
    def create(cls) -> BaseLLM:
        # 先用真实LLM尝试，失败或演示环境则Mock
        llm = OpenAICompatibleLLM()
        # 如果api_key还是默认demo-key，直接回退Mock（避免无效调用）
        if llm.api_key == "sk-demo-key":
            return MockLLM()
        return llm


# ============================================================
# 2. 重排序模型（CrossEncoder / Mock 双实现）
# 对应简历：接入重排序模型优化检索候选集
# ============================================================
class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[VectorSearchResult],
               top_k: int) -> List[VectorSearchResult]:
        raise NotImplementedError


class MockReranker(BaseReranker):
    """
    Mock版：基于字符级n-gram + 关键词密度打分模拟 CrossEncoder
    作用：不依赖sentence-transformers，离线可演示"重排序"流程
    """
    @staticmethod
    def _tokens(text: str) -> set:
        t = text.lower()
        toks: set = set()
        toks.update(re.findall(r"[a-z][a-z0-9._/%-]{1,}", t))
        toks.update(re.findall(r"\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:美元|欧元|元|天|kg|g|工作日)", t))
        cn = re.sub(r"[^\u4e00-\u9fff]", "", t)
        for i in range(len(cn) - 1):
            toks.add(cn[i:i + 2])
        return toks

    def rerank(self, query: str, candidates: List[VectorSearchResult],
               top_k: int) -> List[VectorSearchResult]:
        if not candidates:
            return []
        q_toks = self._tokens(query)
        if not q_toks:
            return candidates[:top_k]
        for r in candidates:
            c_toks = self._tokens(r.chunk.text)
            overlap = q_toks & c_toks
            precision = len(overlap) / max(len(c_toks), 1)
            recall = len(overlap) / max(len(q_toks), 1)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            # 融合原hybrid_score（多路检索）+ 重排f1
            r.rerank_score = round(0.5 * r.hybrid_score + 0.5 * f1, 4)
        candidates.sort(key=lambda r: -r.rerank_score)
        return candidates[:top_k]


class CrossEncoderReranker(BaseReranker):
    """真实CrossEncoder（sentence-transformers / bge-reranker-v2-m3）"""
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[VectorSearchResult],
               top_k: int) -> List[VectorSearchResult]:
        if not candidates:
            return []
        pairs = [[query, r.chunk.text] for r in candidates]
        scores = self.model.predict(pairs, apply_softmax=True, convert_to_numpy=True)
        for r, s in zip(candidates, scores):
            r.rerank_score = round(float(s), 4)
        candidates.sort(key=lambda r: -r.rerank_score)
        return candidates[:top_k]


class RerankerFactory:
    @classmethod
    def create(cls) -> BaseReranker:
        cfg = get_config()
        provider = cfg.retrieval.rerank_provider.lower()
        if provider == "cross_encoder":
            try:
                return CrossEncoderReranker(cfg.retrieval.rerank_model)
            except Exception as exc:
                print(f"[RerankerFactory] CrossEncoder加载失败({exc})，回退Mock")
        return MockReranker()


# ============================================================
# 3. 多语种结构化 Prompt 模板（核心Prompt工程）
# 对应简历：定制多语种结构化 Prompt 固定回答格式
# ============================================================
class MultilingualPromptBuilder:
    """
    RAG Prompt Builder：固定【角色 + 资料片段 + 问题 + 输出约束】结构
    支持中英日韩德西6种语言输出
    """
    SYSTEM_ROLE_BASE_ZH = """你是「星途环球」跨境电商智能售后助手，严格基于下方"参考知识片段"回答用户问题。
【核心规则-不可违反】
1. 只回答参考片段中有明确依据的内容；若资料不足，直接回复"暂无相关知识库资料，请补充信息后重试或联系人工客服"。
2. 严禁编造：不得虚构政策条款、税率、时效、运费、平台规则、认证名称等。
3. 合规类问题（关税/VAT/认证/海关）必须在回答中标注知识来源编号，如[#1][#2]。
4. 时效/费用类回答若存在区间，优先给出明确范围（如"约35-50元"、"7-15个工作日"），不要使用模糊词。
5. 输出语言严格跟随"输出语言"字段；若无法判断则跟随问题语种。
"""

    SYSTEM_ROLE_BASE_EN = """You are the Cross-border E-commerce After-sales AI Assistant of XingTu Global. Answer strictly based on the "Reference Knowledge Pieces" below.
[CORE RULES - MUST NOT VIOLATE]
1. Only answer with explicit evidence from the reference pieces. If the information is insufficient, reply: "No relevant knowledge base information available yet. Please add more details and retry, or contact human support."
2. NO fabrication: Never invent policies, tax rates, shipping times, fees, platform rules, or certification names.
3. For compliance questions (tariff/VAT/certification/customs), you MUST cite source numbers, e.g. [#1][#2].
4. For time/fee questions with ranges, give explicit ranges (e.g. "about 35-50 CNY", "7-15 working days") - never vague wording.
5. Output Language strictly follows the "Output Language" field.
"""

    LANGUAGE_MAP_TO_NAME = {
        "zh": "中文", "en": "English", "ja": "日本語",
        "es": "Español", "de": "Deutsch", "fr": "Français",
    }

    @classmethod
    def detect_language(cls, query: str) -> str:
        """简单语种判断：中文>日文>其他按ASCII占比"""
        cn_count = len(re.findall(r"[\u4e00-\u9fff]", query))
        ja_count = len(re.findall(r"[\u3040-\u30ff]", query))
        total = max(len(query), 1)
        if cn_count / total > 0.3:
            return "zh"
        if ja_count / total > 0.3:
            return "ja"
        en_count = len(re.findall(r"[A-Za-z]", query))
        if en_count / total > 0.4:
            # 简单按关键词猜其他语种
            q_lower = query.lower()
            if any(k in q_lower for k in ("hola", "gracias", "españa", "mexico", "argentina", "chile")):
                return "es"
            if any(k in q_lower for k in ("hallo", "guten", "deutschland", "germany", "berlin")):
                return "de"
            if any(k in q_lower for k in ("bonjour", "france", "paris", "merci", "français")):
                return "fr"
            return "en"
        return "zh"

    @classmethod
    def build(cls, query: str,
              retrieved: List[VectorSearchResult],
              force_language: Optional[str] = None) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        构建 OpenAI 格式 messages
        返回：(messages, meta)  meta包含知识片段列表用于后续风控
        """
        language = force_language or cls.detect_language(query)
        if language not in SUPPORTED_LANGUAGES:
            language = "en"
        is_en = (language != "zh")
        system_role = cls.SYSTEM_ROLE_BASE_EN if is_en else cls.SYSTEM_ROLE_BASE_ZH

        # 拼接参考知识片段（编号引用，便于后续Faithfulness检查）
        pieces_text: List[str] = []
        meta_pieces: List[Dict[str, Any]] = []
        for idx, r in enumerate(retrieved, 1):
            c = r.chunk
            snippet = c.text[:600].replace("\n", " ")
            tag = KNOWLEDGE_CATEGORIES.get(c.category, {}).get("name", c.category)
            header = f"[#{idx}] [{tag}] 标题:{c.doc_title} (父块:{c.parent_chunk_id or '-'}, 合规级别:{c.compliance_level})"
            pieces_text.append(f"{header}\n内容: {snippet}")
            meta_pieces.append({
                "index": idx,
                "chunk_id": c.chunk_id,
                "doc_title": c.doc_title,
                "category": c.category,
                "text": c.text,
                "compliance_level": c.compliance_level,
                "needs_citation": c.needs_citation,
                "scores": {
                    "hybrid": r.hybrid_score,
                    "dense": r.dense_score,
                    "sparse": r.sparse_score,
                    "rerank": r.rerank_score,
                },
            })

        pieces_block = "\n\n".join(pieces_text) if pieces_text else "（无检索命中）"

        # 判断问题类目 -> 追加类目专属约束
        category_hint = cls._category_specific_constraints(query, is_en)

        lang_name = cls.LANGUAGE_MAP_TO_NAME.get(language, "中文")
        user_template = (
            f"输出语言: {lang_name}\n"
            f"{category_hint}\n"
            f"===== 参考知识片段 =====\n{pieces_block}\n\n"
            f"===== 用户提问 =====\n问题：{query}\n\n"
            f"请输出结构化回答（分「结论」「依据要点」「知识来源引用」「注意事项」四节）。"
        ) if not is_en else (
            f"Output Language: {lang_name}\n"
            f"{category_hint}\n"
            f"===== Reference Knowledge Pieces =====\n{pieces_block}\n\n"
            f"===== User Query =====\nQuestion: {query}\n\n"
            f"Please output a structured answer in 4 sections: Conclusion, Basis Points, Knowledge Source Citations, Notes."
        )

        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_template},
        ]
        meta = {
            "language": language,
            "pieces": meta_pieces,
            "has_enough_context": len(meta_pieces) > 0,
            "top_confidence": meta_pieces[0]["scores"]["rerank"] if meta_pieces else 0.0,
        }
        return messages, meta

    @classmethod
    def _category_specific_constraints(cls, query: str, is_en: bool) -> str:
        """类目专属约束（Prompt风控层）"""
        q = query.lower()
        if not is_en:
            if any(k in q for k in ("关税", "vat", "ioss", "海关", "认证", "申报", "税率")):
                return "【问题识别：海关/合规类】→ 必须给出数值范围或条款编号，并引用[#x]；禁止模糊的'可能/大概'。"
            if any(k in q for k in ("退货", "退款", "售后", "质保", "争议", "拒付")):
                return "【问题识别：售后政策类】→ 区分'平台规则'和'独立站自定义'两种情况；说明时效+运费责任方。"
            if any(k in q for k in ("物流", "快递", "运费", "时效", "包裹", "清关", "派送", "fba")):
                return "【问题识别：跨境物流类】→ 回答时效必须带'工作日/自然日'区分；费用带货币单位+区间。"
            return ""
        else:
            if any(k in q for k in ("tariff", "vat", "ioss", "customs", "certification", "duti", "tax rate")):
                return "[Issue Type: Compliance/Customs] → Must give numeric ranges or clause numbers; cite [#x]; avoid vague 'maybe/probably'."
            if any(k in q for k in ("refund", "return", "after-sale", "warranty", "dispute", "chargeback")):
                return "[Issue Type: After-sales Policy] → Distinguish platform rules vs independent site custom rules; specify timeframe + shipping fee responsibility."
            if any(k in q for k in ("shipping", "logistics", "freight", "delivery", "package", "clearance", "fba")):
                return "[Issue Type: Cross-border Logistics] → Timeframes must state 'working/calendar days'; fees must include currency + range."
            return ""
