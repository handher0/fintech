import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Time", page_icon="🕒", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu, header, footer { visibility: hidden; }
    .stApp { background: #000; }
    .block-container { padding-top: 1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 실시간 시계: 검은 배경, 녹색 디지털 시간, 노란색 날짜, 상단 중앙 배치
components.html(
    r"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
    <meta charset="utf-8" />
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
        html, body { margin:0; padding:0; background:#000; }
        .wrap { width:100%; text-align:center; padding-top:24px; }
        .time {
            font-family: 'Orbitron', 'Share Tech Mono', monospace;
            font-size: clamp(48px, 12vw, 120px);
            font-weight: 700;
            color: #00ff41;
            text-shadow: 0 0 12px rgba(0,255,65,.7), 0 0 30px rgba(0,255,65,.4);
            letter-spacing: 4px;
        }
        .date {
            font-family: 'Share Tech Mono', monospace;
            font-size: clamp(20px, 4vw, 40px);
            color: #ffff00;
            margin-top: 12px;
        }
    </style>
    </head>
    <body>
        <div class="wrap">
            <div class="time" id="time">--:--:--</div>
            <div class="date" id="date">----년 --월 --일</div>
        </div>
        <script>
            const days=['일','월','화','수','목','금','토'];
            const p=n=>String(n).padStart(2,'0');
            function tick(){
                const d=new Date();
                document.getElementById('time').textContent =
                    p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds());
                document.getElementById('date').textContent =
                    d.getFullYear()+'년 '+(d.getMonth()+1)+'월 '+d.getDate()+'일 ('+days[d.getDay()]+'요일)';
                setTimeout(tick, 250);
            }
            tick();
        </script>
    </body>
    </html>
    """,
    height=260,
)
