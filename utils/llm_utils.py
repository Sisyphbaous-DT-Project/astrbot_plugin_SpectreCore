from astrbot.api.all import *
from typing import Dict, List, Optional, Any
import time
import threading
from .history_storage import HistoryStorage
from .message_utils import MessageUtils
from astrbot.core.provider.entites import ProviderRequest
from astrbot.core.message.components import Reply
from astrbot.core.utils.quoted_message import extract_quoted_message_images

class LLMUtils:
    """
    大模型调用工具类
    用于构建提示词和调用记录相关功能
    """
    
    # 使用字典保存每个聊天的大模型调用状态
    # 格式: {"{platform_name}_{chat_type}_{chat_id}": {"last_call_time": timestamp, "in_progress": True/False}}
    _llm_call_status: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()  # 用于线程安全的锁
    
    @staticmethod
    def get_chat_key(platform_name: str, is_private_chat: bool, chat_id: str) -> str:
        """
        获取聊天的唯一标识
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            聊天的唯一标识
        """
        chat_type = "private" if is_private_chat else "group"
        return f"{platform_name}_{chat_type}_{chat_id}"
    
    @staticmethod
    def set_llm_in_progress(platform_name: str, is_private_chat: bool, chat_id: str, in_progress: bool = True) -> None:
        """
        设置大模型调用状态
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            in_progress: 是否正在进行大模型调用
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key not in LLMUtils._llm_call_status:
                LLMUtils._llm_call_status[chat_key] = {}
                
            LLMUtils._llm_call_status[chat_key]["in_progress"] = in_progress
            LLMUtils._llm_call_status[chat_key]["last_call_time"] = time.time()
    
    @staticmethod
    def is_llm_in_progress(platform_name: str, is_private_chat: bool, chat_id: str) -> bool:
        """
        检查指定聊天是否正在进行大模型调用
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            是否正在进行大模型调用
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key not in LLMUtils._llm_call_status:
                return False
                
            return LLMUtils._llm_call_status[chat_key].get("in_progress", False)
    
    @staticmethod
    def get_last_call_time(platform_name: str, is_private_chat: bool, chat_id: str) -> Optional[float]:
        """
        获取指定聊天最后一次大模型调用的时间戳
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            最后一次调用的时间戳，如果从未调用过则返回None
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key not in LLMUtils._llm_call_status:
                return None
                
            return LLMUtils._llm_call_status[chat_key].get("last_call_time")
    
    @staticmethod
    async def get_native_conversation(event: AstrMessageEvent, context: Context):
        """获取 AstrBot 原生会话，让影芯触发和普通 @ 回复共用同一份上下文。"""
        try:
            conversation_manager = getattr(context, "conversation_manager", None)
            if conversation_manager is None:
                return None

            umo = event.unified_msg_origin
            conversation_id = await conversation_manager.get_curr_conversation_id(umo)
            if not conversation_id:
                conversation_id = await conversation_manager.new_conversation(
                    umo,
                    event.get_platform_id(),
                )

            conversation = await conversation_manager.get_conversation(
                umo,
                conversation_id,
            )
            if conversation is None:
                conversation_id = await conversation_manager.new_conversation(
                    umo,
                    event.get_platform_id(),
                )
                conversation = await conversation_manager.get_conversation(
                    umo,
                    conversation_id,
                )
            return conversation
        except Exception as e:
            logger.warning(f"获取 AstrBot 原生会话失败，将回退影芯轻量上下文: {e}")
            return None

    @staticmethod
    async def build_legacy_persona_context(context: Context, umo: str):
        """原有影芯轻量人格上下文兜底，仅在拿不到 AstrBot 原生会话时使用。"""
        system_prompt = ""
        contexts = []

        try:
            # 遵循 AstrBot 原生人格获取优先级：
            # 1. session_service_config.persona_id (会话级别配置)
            # 2. 配置文件的 default_personality
            # 3. 全局默认人格

            persona_id = None
            persona = None

            # 优先级1: 查询会话级别的人格配置 (通过全局 SharedPreferences)
            try:
                from astrbot.api import sp
                session_config = await sp.get_async(
                    scope="umo", scope_id=umo, key="session_service_config", default={}
                )
                persona_id = session_config.get("persona_id")
                if persona_id:
                    logger.debug(f"从 session_service_config 获取人格: '{persona_id}'")
            except Exception as e:
                logger.debug(f"获取 session_service_config 失败: {e}")

            # 优先级2/3: 使用 get_default_persona_v3 获取配置文件或全局默认人格
            if not persona_id:
                if hasattr(context, 'persona_manager') and hasattr(context.persona_manager, 'get_default_persona_v3'):
                    persona = await context.persona_manager.get_default_persona_v3(umo=umo)
                    persona_id = persona.get('name') if persona else None
                else:
                    # Fallback: 旧版 AstrBot 兼容
                    persona = context.persona_manager.selected_default_persona_v3 if hasattr(context, 'persona_manager') else None
                    persona_id = persona.get('name') if persona else None

            # 根据 persona_id 获取完整的人格数据
            if persona_id and not persona:
                try:
                    persona = next(
                        (p for p in context.persona_manager.personas_v3 if p["name"] == persona_id),
                        None
                    )
                except Exception:
                    pass

            if persona:
                system_prompt = persona.get('prompt', '')
                if persona.get('_mood_imitation_dialogs_processed'):
                    mood_dialogs = persona.get('_mood_imitation_dialogs_processed', '')
                    if mood_dialogs:
                        system_prompt += "\n请模仿以下示例的对话风格来反应(示例中，a代表用户，b代表你)\n" + str(mood_dialogs)

                begin_dialogs = persona.get('_begin_dialogs_processed', [])
                if begin_dialogs:
                    contexts.extend(begin_dialogs)

                logger.debug(f"使用 UMO '{umo}' 对应的人格: '{persona.get('name', 'default')}'")
        except Exception as e:
            logger.error(f"获取人格信息失败: {e}")

        return system_prompt, contexts

    @staticmethod
    async def collect_quoted_image_urls(event: AstrMessageEvent) -> List[str]:
        """收集当前消息引用中的图片，补齐主动回复链路对引用图的视觉输入。"""
        image_urls: List[str] = []
        try:
            message_components = getattr(getattr(event, "message_obj", None), "message", [])
            for component in message_components:
                if not isinstance(component, Reply):
                    continue
                try:
                    quoted_images = await extract_quoted_message_images(event, component)
                except Exception as e:
                    logger.warning(f"解析引用图片失败: {e}")
                    continue

                for url in quoted_images or []:
                    if url and url not in image_urls:
                        image_urls.append(url)
        except Exception as e:
            logger.warning(f"收集引用图片URL时出错: {e}")

        return image_urls

    @staticmethod
    async def call_llm(event: AstrMessageEvent, config: AstrBotConfig, context: Context) -> ProviderRequest:
        """
        构建调用大模型的请求对象

        Args:
            event: 消息对象
            config: 配置对象
            context: Context 对象，用于获取LLM工具管理器

        Returns:
            ProviderRequest 对象
        """
        platform_name = event.get_platform_name()
        is_private = event.is_private_chat()
        chat_id = event.get_group_id() if not is_private else event.get_sender_id()

        # 准备并调用大模型
        func_tools_mgr = context.get_llm_tool_manager() if config.get("use_func_tool", False) else None

        # 优先使用 AstrBot 原生会话：人格、工具、主会话历史和截断规则都交给 AstrBot 处理。
        system_prompt = ""
        contexts = []
        umo = event.unified_msg_origin
        conversation = await LLMUtils.get_native_conversation(event, context)
        if conversation is not None:
            logger.debug(f"使用 AstrBot 原生会话上下文: umo={umo}")
        else:
            system_prompt, contexts = await LLMUtils.build_legacy_persona_context(
                context,
                umo,
            )

        # 构建环境描述（注入到 system_prompt，不污染 prompt）
        env_description = f"\n\n你正在浏览聊天软件，你在聊天软件上的id是{event.get_self_id()}"

        # 对于aiocqhttp平台，尝试获取bot用户名
        if platform_name == "aiocqhttp" and hasattr(event, "bot"):
            try:
                bot = getattr(event, "bot")
                bot_name = (await bot.api.get_login_info())["nickname"]
                env_description += f"，用户名是{bot_name}"
            except Exception as e:
                logger.warning(f"通过 event.bot 获取机器人昵称失败: {e}")

        if is_private:
            sender_display_name = event.get_sender_name() if event.get_sender_name() else f"ID为 {event.get_sender_id()} 的人"
            env_description += f"，你正在和 {sender_display_name} 私聊页面中。"
        else:
            group_display_name = chat_id
            if platform_name in ["aiocqhttp", "gewechat"]:
                try:
                    group = await event.get_group()
                    if group and group.group_name:
                        group_display_name = f"{group.group_name}({chat_id})"
                except Exception as e:
                    logger.warning(f"为 {platform_name} 获取群组信息失败: {e}")
            env_description += f"，你正在群聊 {group_display_name} 中。"

        # 添加历史记录（文本格式，注入到 system_prompt）
        # 注意：基于 message_id 精确排除当前消息，避免重复
        history_limit = config.get("group_msg_history", 10)
        try:
            history_limit = int(history_limit)
        except (TypeError, ValueError):
            history_limit = 10

        history_messages = []
        current_msg_id = getattr(event.message_obj, 'message_id', None) if hasattr(event, 'message_obj') else None
        if history_limit > 0:
            history_messages = HistoryStorage.get_history(platform_name, is_private, chat_id)

            try:
                if conversation is not None:
                    logger.debug("已跳过影芯自带聊天记录注入，改用 AstrBot 原生 conversation history")
                elif history_messages:
                    # 获取当前消息的 message_id 用于精确排除
                    if current_msg_id:
                        history_for_context = [m for m in history_messages if getattr(m, 'message_id', None) != current_msg_id]
                    else:
                        # 回退到排除最后一条
                        history_for_context = history_messages[:-1] if len(history_messages) > 1 else []
                    if history_for_context:
                        formatted_history = await MessageUtils.format_history_for_llm(history_for_context, max_messages=history_limit, umo=umo)
                        env_description += "\n\n以下是最近的聊天记录：\n" + formatted_history
                    else:
                        env_description += "\n\n你没看见任何聊天记录，看来最近没有消息。"
                else:
                    env_description += "\n\n你没看见任何聊天记录，看来最近没有消息。"
            except Exception as e:
                logger.error(f"获取或格式化历史记录失败: {e}")
                env_description += "\n\n你没看见任何聊天记录，看来最近没有消息。"

        # 行为指引
        env_description += "\n(在聊天记录中，你的用户名以AstrBot被代替了)"
        env_description += "\n(如果你想回复某人，不要使用类似 [At:id(昵称)]这样的格式)"

        if config.get("read_air", False):
            env_description += "\n\n现在你收到了一条新消息，你的反应是:\n(如果你想发送一条消息，直接输出发送的内容，如果你选择忽略，直接输出<NO_RESPONSE>)"
        else:
            env_description += "\n\n现在你收到了一条新消息，你决定发送一条消息回复(你输出的内容将作为消息发送)"

        # 将环境描述追加到 system_prompt
        system_prompt += env_description

        # 图片相关处理
        image_urls = []

        # 首先收集当前消息链中的图片（用户刚发送的，不受 image_count 限制）
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
            for component in event.message_obj.message:
                if isinstance(component, Image):
                    try:
                        url = component.file or component.url
                        if url and url not in image_urls:
                            image_urls.append(url)
                    except Exception as e:
                        logger.warning(f"处理当前消息图片URL时出错: {e}")
                        continue

            quoted_image_urls = await LLMUtils.collect_quoted_image_urls(event)
            for url in quoted_image_urls:
                if url and url not in image_urls:
                    image_urls.append(url)
            if quoted_image_urls:
                system_prompt += f"\n\n当前消息引用了{len(quoted_image_urls)}张图片，已经作为本轮视觉输入提供给你。"

        # 然后从历史消息中补充收集图片（受 image_count 限制）
        history_image_count = config.get("image_processing", {}).get("image_count", 0)
        if history_image_count and history_messages:
            messages_to_show = history_messages[-history_limit:] if len(history_messages) > history_limit else history_messages
            history_image_added = 0

            for message in reversed(messages_to_show):
                if current_msg_id and getattr(message, 'message_id', None) == current_msg_id:
                    continue
                if hasattr(message, "message") and message.message:
                    for component in message.message:
                        if isinstance(component, Image):
                            try:
                                url = component.file or component.url
                                if url and url not in image_urls:
                                    image_urls.append(url)
                                    history_image_added += 1
                                    if history_image_added >= history_image_count:
                                        break
                            except Exception as e:
                                logger.warning(f"处理历史消息图片URL时出错: {e}")
                                continue
                    if history_image_added >= history_image_count:
                        break

            if history_image_added:
                system_prompt += f"\n\n已经按照从晚到早的顺序为你提供了聊天记录中的{history_image_added}张图片，你可以直接查看并理解它们。这些图片出现在聊天记录中。"

        # prompt 只保留用户当前消息，使用 MessageUtils 确保图片被转述
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
            prompt = await MessageUtils.outline_message_list(event.message_obj.message, umo=umo)
        else:
            prompt = event.get_message_outline()

        return event.request_llm(
            prompt=prompt,
            func_tool_manager=func_tools_mgr,
            contexts=contexts,
            system_prompt=system_prompt,
            image_urls=image_urls,
            conversation=conversation,
        )
    
    @staticmethod
    def clear_call_status(platform_name: str, is_private_chat: bool, chat_id: str) -> None:
        """
        清除指定聊天的大模型调用状态
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key in LLMUtils._llm_call_status:
                del LLMUtils._llm_call_status[chat_key] 
