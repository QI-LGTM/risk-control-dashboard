
import streamlit as st

# 1. 页面基本配置（必须是Streamlit命令的第一句）
st.set_page_config(
    page_title="金融风控大作业展示",
    page_icon="🛡️",
    layout="wide",  # 宽屏模式，看图表更舒服
    initial_sidebar_state="expanded"
)

# 2. 侧边栏内容
st.sidebar.markdown("# 🧭 全局导航")
st.sidebar.info("请在上方选择具体的风控任务查看实验成果。")

# 3. 主页面内容
st.title("🛡️ 金融风控大数据赛事实验成果仪表盘")
st.markdown("---")

# 4. 用两列布局展示两个任务的简介
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/Give_Me_Some_Credit.py", label="📈 任务 1: Give Me Some Credit", use_container_width=True)

    st.markdown("""
        **任务描述：** 基于传统信用数据，预测借款人在未来两年内是否会陷入财务困境。
        """)
    st.caption("✨ 点击上方标题即可跳转查看详情")

with col2:
    st.page_link("pages/Home_Credit_Default_Risk.py", label="🏢 任务 2: Home Credit Default Risk",
                 use_container_width=True)

    st.markdown("""
        **任务描述：** 预测客户贷款违约概率，构建有效的风控模型。
        """)
    st.caption("✨ 点击上方标题即可跳转查看详情")

st.markdown("---")
st.subheader("👤 学生信息与实验说明")
st.info("本大作业严格按照要求实现了两个方向的完整数据科学流水线，包含算法对比与丰富的可视化成果。")
