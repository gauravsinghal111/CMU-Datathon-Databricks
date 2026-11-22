"""
VeriCare - find the right facility for a service, with the evidence behind every answer.

Reads the tables built by the notebooks:
  facility_service_evidence (Notebook 2), pin_lookup, place_lookup, cleaning_report (Notebook 1),
  size_model_* (Notebook 4), care_gaps_* (Notebook 6), india_* + facility_contacts (Notebook 7).
The app itself never calls Google Maps.

Local testing: set VERICARE_LOCAL_DIR to a folder of <table>.parquet files and run  streamlit run app.py
"""
import json
import math
import os
import re
import time

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

st.set_page_config(page_title="VeriCare", page_icon=":material/verified_user:", layout="wide", initial_sidebar_state="collapsed")
DB = "workspace.default"
SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
LOCAL_DIR = os.environ.get("VERICARE_LOCAL_DIR", "").strip()

# ---------------------------------------------------------------- design system
INK, MUTED, LINE = "#1c1210", "#7a6a64", "#f1e4dd"
BRAND, BRAND2 = "#ef4f3c", "#b8325a"            # coral, berry
GREEN, AMBER, RED, GREY = "#2f9e6e", "#e9a23b", "#dc3d43", "#bcaea8"
REGION_COLOR = {"EAG states": RED, "South": GREEN, "Rest of India": GREY}

LOGO = ('<svg width="{s}" height="{s}" viewBox="0 0 24 24" aria-hidden="true">'
        '<path d="M12 2l8 3v6c0 5-3.4 9.4-8 11-4.6-1.6-8-6-8-11V5l8-3z" fill="white" fill-opacity=".95"/>'
        '<path class="{cls}" d="M8 12.5l2.6 2.6L16.5 9" stroke="#ef4f3c" stroke-width="2.4" fill="none" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>')

_ICON_PATHS = {
    "phone": '<path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 '
             '2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1.9.4 1.8.7 2.7a2 2 0 0 1-.5 2.1L8 9.8a16 16 0 0 0 6 6l1.3-1.3a2 2 0 '
             '0 1 2.1-.4c.9.3 1.8.6 2.7.7a2 2 0 0 1 1.7 2z"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 '
             '15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    "mail": '<rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-10 6L2 7"/>',
    "bed": '<path d="M2 4v16M2 8h18a2 2 0 0 1 2 2v10M2 17h20M6 8v9"/>',
    "user": '<circle cx="12" cy="8" r="5"/><path d="M20 21a8 8 0 0 0-16 0"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0z"/><circle cx="12" cy="10" r="3"/>',
    "alert": '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 '
             '2 0 0 0-3.4 0z"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "map": '<path d="M9 3 3 6v15l6-3 6 3 6-3V3l-6 3-6-3zM9 3v15M15 6v15"/>',
    "pointer": '<path d="m4 4 7 17 2.5-7.5L21 11z"/>',
}


def icon(name, size=15):
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_ICON_PATHS[name]}</svg>')


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
:root {--coral: #ef4f3c; --berry: #b8325a; --peach: #f6b896; --ink: #1c1210; --muted: #7a6a64; --line: #f1e4dd;
       --cream: #fbf5f1; --display: 'Inter Tight', 'Inter', system-ui, sans-serif;}
html, body, [class*="css"], .stMarkdown, button, input, textarea, select {font-family: 'Inter', system-ui, sans-serif;}
html, body, .stApp {overflow-x: hidden;}
.stApp {background: var(--cream); color: var(--ink);}
header[data-testid="stHeader"] {background: transparent; height: 0;}
[data-testid="stToolbar"], footer, [data-testid="stDecoration"] {display: none !important;}
.block-container {padding-top: 0 !important; padding-bottom: 4rem; max-width: 1380px;}
h1, h2, h3, h4 {font-family: var(--display); letter-spacing: -0.03em; color: var(--ink);}
.ic {display: inline-block; vertical-align: -2px; margin-right: 6px; flex: none;}

/* ---------- motion */
@keyframes fadeUp {from {opacity: 0; transform: translateY(18px);} to {opacity: 1; transform: none;}}
@keyframes fadeIn {from {opacity: 0;} to {opacity: 1;}}
@keyframes growX {from {transform: scaleX(0);} to {transform: scaleX(1);}}
@keyframes rise {from {opacity: 0; transform: translateY(26px);} to {opacity: 1; transform: none;}}
@keyframes floaty {0%, 100% {transform: translate3d(0, 0, 0) rotateX(var(--rx)) rotateY(var(--ry)) rotateZ(var(--rz));}
                   50% {transform: translate3d(0, -22px, 0) rotateX(var(--rx)) rotateY(calc(var(--ry) + 14deg)) rotateZ(var(--rz));}}
@keyframes splashOut {to {opacity: 0; visibility: hidden;}}
@keyframes bar {from {width: 0;} to {width: 100%;}}
@property --n {syntax: '<integer>'; initial-value: 0; inherits: false;}
@keyframes count {to {--n: var(--to);}}

/* ---------- hero (full-bleed gradient, giant type, floating discs) */
.hero {position: relative; width: 100vw; left: 50%; margin-left: -50vw; overflow: hidden; color: white;
       background: radial-gradient(120% 90% at 0% 100%, #d9364f 0%, rgba(217,54,79,0) 55%),
                   linear-gradient(118deg, #e0445e 0%, #ee6a5c 38%, #f49a7a 70%, #f7c6a2 100%);
       padding: 26px 5vw 54px 5vw; margin-bottom: 34px;}
.hero.big {min-height: 76vh; display: flex; flex-direction: column; justify-content: space-between;}
.hero.small {min-height: 250px; display: flex; flex-direction: column; justify-content: space-between; padding-bottom: 34px;}
.hero .brand {position: relative; z-index: 2; display: flex; align-items: center; gap: 10px; font-family: var(--display);
              font-weight: 700; font-size: 1.9rem; letter-spacing: -0.04em; animation: fadeIn .8s ease both;}
.hero .display {position: relative; z-index: 2; font-family: var(--display); font-weight: 600; color: white;
                letter-spacing: -0.055em; line-height: .92; margin: 0; animation: rise 1s cubic-bezier(.2,.8,.2,1) .1s both;}
.hero.big .display {font-size: clamp(3.4rem, 8.2vw, 8.4rem);}
.hero.small .display {font-size: clamp(2.6rem, 5.2vw, 4.8rem);}
.hero .bottom {position: relative; z-index: 2; display: flex; align-items: flex-end; justify-content: space-between;
               gap: 40px; flex-wrap: wrap;}
.hero .side {max-width: 420px; animation: rise 1s cubic-bezier(.2,.8,.2,1) .3s both;}
.hero .side p {font-family: var(--display); font-size: 1.45rem; font-weight: 600; line-height: 1.2; letter-spacing: -0.02em;
               margin: 0 0 18px 0; color: white;}
.hero .stats {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px;}
/* longer scripts (Tamil, Malayalam, ...) need smaller, airier display type */
.hero.big.intl .display {font-size: clamp(2.6rem, 4.6vw, 4.6rem); line-height: 1.12; letter-spacing: -0.02em;}
.hero.small.intl .display {font-size: clamp(2rem, 3.6vw, 3.4rem); line-height: 1.15; letter-spacing: -0.02em;}
.hero.intl .side p {font-size: 1.15rem; line-height: 1.4;}
.hero.intl .bottom {flex-wrap: nowrap;} .hero.intl .display {flex: 1 1 auto; min-width: 0;}
.hero.intl .side {flex: 0 0 400px;}
.hero h1 a, .hero [data-testid="stHeaderActionElements"], [data-testid="stHeaderActionElements"] {display: none !important;}
.hero .stat {border: 1.5px solid rgba(255,255,255,.55); border-radius: 16px; padding: 9px 14px; backdrop-filter: blur(6px);
             background: rgba(255,255,255,.08);}
.hero .stat b {display: block; font-family: var(--display); font-size: 1.25rem; letter-spacing: -0.02em;}
.hero .stat span {font-size: .74rem; opacity: .9;}
.discs {position: absolute; inset: 0; perspective: 1400px; pointer-events: none; z-index: 1;}
.d {position: absolute; border-radius: 50%; transform-style: preserve-3d; animation: floaty var(--t, 9s) ease-in-out infinite;
    background: linear-gradient(140deg, #ffc9b8 0%, #f47a68 35%, #e2475c 70%, #c93556 100%);
    box-shadow: inset 0 0 0 2px rgba(255,255,255,.45), inset -26px -30px 60px rgba(120,16,40,.32),
                inset 24px 26px 50px rgba(255,230,215,.35), 0 50px 90px -40px rgba(120,20,40,.6);}
.d1 {width: 360px; height: 360px; right: 30%; top: -12%; --rx: 58deg; --ry: -18deg; --rz: -30deg; --t: 10s;}
.d2 {width: 280px; height: 280px; right: 45%; top: 12%; --rx: 64deg; --ry: 12deg; --rz: -38deg; --t: 8s; animation-delay: -2s;}
.d3 {width: 320px; height: 320px; right: 17%; top: 14%; --rx: 70deg; --ry: -8deg; --rz: -24deg; --t: 11s; animation-delay: -4s;}
.d4 {width: 220px; height: 220px; right: 40%; top: 50%; --rx: 60deg; --ry: 20deg; --rz: -44deg; --t: 9s; animation-delay: -1s;}
.d5 {width: 160px; height: 160px; right: 4%; top: 2%; --rx: 52deg; --ry: -24deg; --rz: -20deg; --t: 12s; animation-delay: -6s; opacity: .85;}
.hero.small .d {transform-origin: center; scale: .55;}
.hero.small .d1 {top: -60%;} .hero.small .d2 {top: -10%;} .hero.small .d3 {top: 0;} .hero.small .d4 {display: none;}

/* ---------- floating pill navigation + language pill */
[class*="st-key-view_"] {position: fixed !important; top: 18px; left: 50%; transform: translateX(-50%); z-index: 999990;
                         width: auto !important; animation: fadeIn .8s ease .2s both;}
[class*="st-key-view_"] [role="radiogroup"] {background: var(--coral) !important; border: none !important; padding: 5px !important;
    border-radius: 999px !important; gap: 2px !important; box-shadow: 0 14px 36px -14px rgba(150,30,30,.55);}
[class*="st-key-view_"] [role="radiogroup"] {flex-wrap: nowrap !important; white-space: nowrap;}
[class*="st-key-view_"] [role="radiogroup"] label {padding: 9px 20px !important; border-radius: 999px !important;}
[class*="st-key-view_"]:not(.st-key-view_English) [role="radiogroup"] label {padding: 9px 14px !important;}
[class*="st-key-view_"]:not(.st-key-view_English) [role="radiogroup"] label p {font-size: .86rem;}
[class*="st-key-view_"] [role="radiogroup"] label p {color: white !important; font-weight: 600; font-size: .95rem;}
[class*="st-key-view_"] [role="radiogroup"] label:hover {background: rgba(255,255,255,.14) !important;}
[class*="st-key-view_"] [role="radiogroup"] label:has(input:checked) {background: rgba(255,255,255,.24) !important;
    box-shadow: none !important;}
/* invisible helpers (style block, splash, nav row) take no space, so the hero starts at the very top */
[data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] style),
[data-testid="stElementContainer"]:has(#vc-splash),
[data-testid="stLayoutWrapper"]:has([class*="st-key-view_"]),
[data-testid="stHorizontalBlock"]:has([class*="st-key-view_"]) {position: absolute !important; height: 0; min-height: 0;
    margin: 0; padding: 0; overflow: visible;}
.st-key-lang {position: fixed !important; top: 18px; right: 28px; width: 150px !important; z-index: 999990;}
div.st-key-lang div[data-baseweb="select"], div.st-key-lang [data-baseweb="select"] * {border-radius: 999px !important;}
div.st-key-lang div[data-baseweb="select"] > div {border-radius: 999px !important; padding-left: 8px; background: rgba(255,255,255,.92) !important;
    border: none !important; box-shadow: 0 10px 30px -12px rgba(120,30,30,.45); font-weight: 600;}

/* ---------- segmented controls (other radios) */
div[data-testid="stRadio"] [role="radiogroup"] {gap: 2px; background: white; padding: 4px; border-radius: 999px;
    display: inline-flex; flex-wrap: wrap; border: 1px solid var(--line); width: auto;}
div[data-testid="stRadio"] [role="radiogroup"] label {margin: 0 !important; padding: 7px 18px; border-radius: 999px;
    transition: background .25s, color .25s; cursor: pointer; color: var(--muted);}
div[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child,
div[data-testid="stRadio"] [role="radiogroup"] label > div > div:not([data-testid="stMarkdownContainer"]) {display: none !important;}
div[data-testid="stRadio"] [role="radiogroup"] label:hover {background: #fdf0ea;}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {background: var(--ink);}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {color: white; font-weight: 600;}

/* ---------- cards & panels */
div[data-testid="stVerticalBlockBorderWrapper"] {animation: fadeUp .6s cubic-bezier(.2,.8,.2,1) both;
    border-radius: 26px !important; background: white; border: 1px solid var(--line) !important;
    box-shadow: 0 1px 2px rgba(60,20,10,.04), 0 24px 48px -32px rgba(120,40,30,.28); transition: box-shadow .3s, transform .3s;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {box-shadow: 0 2px 4px rgba(60,20,10,.05), 0 30px 60px -30px rgba(200,70,50,.35);}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {animation: fadeIn .7s ease both;}
[data-testid="stChatMessage"] {animation: rise .5s ease both; background: transparent;}
[data-testid="stPlotlyChart"] .bars .point path {transform-box: fill-box; transform-origin: left center;
    animation: growX .9s cubic-bezier(.2,.8,.2,1) both;}
[data-testid="stPlotlyChart"] .scatterlayer .point {animation: fadeIn .8s ease both;}
%STAGGER%

.pill {display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 999px; font-size: .78rem; font-weight: 600;
       margin: 0 6px 6px 0; white-space: nowrap;}
.pill-HIGH {background: #e3f4ec; color: #1f7a52;} .pill-MEDIUM {background: #fdf0d9; color: #94600f;}
.pill-LOW {background: #fde3e2; color: #a52a2f;} .pill-grey {background: #f6eeea; color: #5b4a44;}
.pill-google {background: #fde6df; color: #b23a2a;}
.rank {display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 50%;
       color: white; font-family: var(--display); font-weight: 700; margin-right: 12px; font-size: .95rem;}
.fname {font-family: var(--display); font-size: 1.22rem; font-weight: 700; letter-spacing: -0.02em; color: var(--ink);}
.addr {color: var(--muted); font-size: .86rem; margin: 4px 0 10px 0;}
.quote {font-size: .92rem; margin: 4px 0; color: #3a2b27; border-left: 2px solid var(--peach); padding-left: 10px;}
.quote b {color: var(--muted); font-weight: 600;}
.contact {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; font-size: .86rem;}
.contact a, .contact span.c {display: inline-flex; align-items: center; background: var(--cream); border: 1px solid var(--line);
       border-radius: 999px; padding: 6px 13px; color: var(--ink); text-decoration: none; transition: all .2s;}
.contact a:hover {border-color: var(--coral); background: #fff1ec; color: var(--coral);}
.info {position: relative; display: inline-flex; align-items: center; justify-content: center; width: 17px; height: 17px;
       border-radius: 50%; background: #fde6df; color: var(--coral); font-size: .68rem; font-weight: 800; font-style: normal;
       margin-left: 6px; cursor: help; outline: none; transition: background .2s, color .2s;}
.info:hover, .info:focus {background: var(--coral); color: white;}
.info .tip {position: absolute; bottom: calc(100% + 10px); left: 50%; width: 270px; padding: 12px 14px; z-index: 9999;
            background: var(--ink); color: #f3e8e3; border-radius: 14px; font-size: .8rem; font-weight: 400; line-height: 1.45;
            text-align: left; white-space: normal; box-shadow: 0 18px 40px -12px rgba(40,10,5,.55);
            opacity: 0; visibility: hidden; pointer-events: none; transform: translate(-50%, 6px) scale(.97);
            transition: opacity .18s ease, transform .18s ease, visibility .18s;}
.info .tip::after {content: ""; position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
                   border: 7px solid transparent; border-top-color: var(--ink);}
.info .tip b {display: block; font-size: .84rem; margin-bottom: 3px; color: white;}
.info .tip i {display: block; margin-top: 6px; color: #bfa9a1; font-style: normal; font-size: .74rem;}
.info:hover .tip, .info:focus .tip {opacity: 1; visibility: visible; transform: translate(-50%, 0) scale(1);}
.summary {display: flex; align-items: center; flex-wrap: wrap; gap: 6px; font-size: .95rem; color: #3a2b27; margin: 6px 0 12px 0;}
.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 8px 0 18px 0;}
.kpi {background: white; border: 1px solid var(--line); border-radius: 22px; padding: 18px 20px; animation: fadeUp .6s ease both;}
.kpi .v {font-family: var(--display); font-size: 2.3rem; font-weight: 700; letter-spacing: -0.04em;}
.kpi .l {font-size: .82rem; color: var(--muted); margin-top: 2px;}
.count {animation: count 1.4s cubic-bezier(.2,.8,.2,1) forwards; counter-reset: n var(--n);}
.count::after {content: counter(n) "%";}
.hint {min-height: 180px; display: flex; align-items: center; justify-content: center; gap: 8px; text-align: center;
       color: var(--muted); font-size: .92rem; border: 1.5px dashed #ecd6cc; border-radius: 22px; padding: 24px; margin-top: 30px;
       background: #fffaf7; animation: fadeIn .6s both;}
.ptitle {font-family: var(--display); font-weight: 700; font-size: 1.2rem; letter-spacing: -0.02em; color: var(--ink);
         overflow-wrap: anywhere;}
.psub {color: var(--muted); font-size: .82rem; margin-bottom: 8px;}
.kv {display: flex; flex-direction: column; font-size: .88rem; margin: 4px 0 10px 0;}
.kv .r {display: flex; justify-content: space-between; align-items: baseline; gap: 14px; padding: 7px 0;
        border-bottom: 1px solid #f6ece7; animation: fadeIn .4s ease both;}
.kv .r span {color: var(--muted); flex: 1 1 50%; min-width: 0; overflow-wrap: anywhere;}
.kv .r b {color: var(--ink); text-align: right; flex: 1 1 50%; min-width: 0; overflow-wrap: anywhere; font-weight: 650;}
.ins-n {display: inline-flex; width: 32px; height: 32px; border-radius: 50%; align-items: center; justify-content: center;
        background: var(--coral); color: white; font-family: var(--display); font-weight: 700; margin-right: 12px; font-size: .9rem;}
.ins-h {font-family: var(--display); font-size: 1.35rem; font-weight: 700; letter-spacing: -0.025em; color: var(--ink);}
.ins-d {color: #4a3a35; font-size: .95rem; margin: 8px 0 6px 0; line-height: 1.55;}
.section {font-family: var(--display); font-size: 2.1rem; font-weight: 700; letter-spacing: -0.04em; margin: 26px 0 4px 0;
          color: var(--ink);}
.searchline {font-size: .84rem; color: var(--muted); margin-bottom: 6px;}
/* only the Find-care map column follows you down the page (panels scroll normally) */
div[data-testid="stColumn"]:nth-of-type(2):has(iframe), div[data-testid="column"]:nth-of-type(2):has(iframe) {
    position: sticky; top: 5rem; align-self: flex-start;}
div[data-testid="stColumn"] {min-width: 0;}
.stButton > button {border-radius: 999px; border: 1px solid var(--line); background: white; transition: all .2s;
                    padding: 8px 18px; color: var(--ink);}
.stButton > button:hover {border-color: var(--coral); color: var(--coral); transform: translateY(-1px);}
[data-baseweb="select"] > div, [data-testid="stTextInput"] input {border-radius: 14px !important;}
[data-testid="stExpander"] details {border-radius: 18px !important; border-color: var(--line) !important; background: white;}
iframe {border-radius: 22px;}
[data-testid="stChatInput"] {border-radius: 999px !important;}

/* ---------- splash */
#vc-splash {position: fixed; inset: 0; z-index: 1000000; display: flex; flex-direction: column; align-items: flex-start;
            justify-content: flex-end; padding: 0 6vw 12vh 6vw; overflow: hidden;
            background: radial-gradient(120% 90% at 0% 100%, #d9364f 0%, rgba(217,54,79,0) 55%),
                        linear-gradient(118deg, #e0445e 0%, #ee6a5c 38%, #f49a7a 70%, #f7c6a2 100%);
            animation: splashOut .8s ease 2.2s forwards; pointer-events: none;}
#vc-splash .word {position: relative; z-index: 2; font-family: var(--display); font-weight: 600; color: white;
                  font-size: clamp(4rem, 13vw, 12rem); letter-spacing: -0.06em; line-height: .9;
                  animation: rise 1s cubic-bezier(.2,.8,.2,1) .15s both;}
#vc-splash .sub {position: relative; z-index: 2; color: white; font-family: var(--display); font-size: 1.4rem; font-weight: 500;
                 margin-top: 14px; opacity: .95; animation: rise 1s cubic-bezier(.2,.8,.2,1) .4s both;}
#vc-splash .load {position: relative; z-index: 2; width: 220px; height: 3px; background: rgba(255,255,255,.3);
                  border-radius: 3px; margin-top: 26px; overflow: hidden;}
#vc-splash .load i {display: block; height: 100%; background: white; animation: bar 2s ease both;}
</style>
"""
DISCS = '<div class="discs"><i class="d d1"></i><i class="d d2"></i><i class="d d3"></i><i class="d d4"></i><i class="d d5"></i></div>'
STAGGER = "\n".join(
    f"[data-testid=\"stPlotlyChart\"] .bars .point:nth-child({i}) path, "
    f"[data-testid=\"stPlotlyChart\"] .scatterlayer .point:nth-child({i}) {{animation-delay: {0.025 * i:.3f}s;}}"
    for i in range(2, 41))
st.markdown(CSS.replace("%STAGGER%", STAGGER), unsafe_allow_html=True)

if not st.session_state.get("splash_done"):
    st.session_state.splash_done = True
    st.markdown(f'<div id="vc-splash">{DISCS}<div class="word">VeriCare</div>'
                f'<div class="sub">The right care, and the evidence to trust it.</div><div class="load"><i></i></div></div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- data access
@st.cache_resource
def get_conn():
    from databricks import sql
    from databricks.sdk.core import Config
    cfg = Config()
    return sql.connect(server_hostname=cfg.host,
                       http_path=f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}",
                       credentials_provider=lambda: cfg.authenticate)


def query(q):
    if LOCAL_DIR:                                   # local testing: one parquet file per table
        name = re.search(r"FROM\s+([\w.]+)", q, re.I).group(1).split(".")[-1]
        path = os.path.join(LOCAL_DIR, f"{name}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pd.read_parquet(path)
    with get_conn().cursor() as c:
        c.execute(q)
        return c.fetchall_arrow().to_pandas()


def optional(table, src=DB):
    try:
        return query(f"SELECT * FROM {src}.{table}")
    except Exception:
        return pd.DataFrame()


def as_list(v):
    if isinstance(v, str) and v.startswith("["):
        try:
            return json.loads(v)
        except Exception:
            return []
    if isinstance(v, np.ndarray):
        return [list(x) if isinstance(x, np.ndarray) else x for x in v.tolist()]
    return v if isinstance(v, list) else []


EMAIL_RX = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}", re.I)


def emails_of(v):
    out = []
    for e in EMAIL_RX.findall(str(v or "")):
        e = e.strip(".").lower()
        if not re.search(r"\.(png|jpe?g|gif|webp|svg|html?|php)$", e) and e not in out:
            out.append(e)
    return out


def load_contacts():
    """unique_id -> email, official website. From Notebook 7's copy, else straight from the original dataset."""
    c = optional("facility_contacts")
    if not len(c):
        try:
            c = query(f"SELECT unique_id, email, officialWebsite AS official_website FROM {SRC}.facilities")
        except Exception:
            return pd.DataFrame(columns=["unique_id", "email", "official_website"])
    c["email"] = c["email"].map(lambda v: ", ".join(emails_of(v)[:2]) or None) if "email" in c else None
    if "official_website" not in c:
        c["official_website"] = None
    return c[["unique_id", "email", "official_website"]].drop_duplicates("unique_id")


@st.cache_resource(ttl=3600, show_spinner="Loading VeriCare…")
def load_all():
    ev = query(f"SELECT * FROM {DB}.facility_service_evidence")
    for c in ["quotes", "negated_quotes", "warnings", "source_urls", "checks", "aliases"]:
        ev[c] = ev[c].map(as_list) if c in ev.columns else [[] for _ in range(len(ev))]
    for c in ["google_match", "g_name", "g_rating", "g_reviews_count", "g_maps_url", "website", "phone",
              "beds_value", "beds_source", "beds_low", "beds_high", "beds_p_large", "beds_class", "beds_class_shown",
              "doctors_value", "doctors_source", "doctors_low", "doctors_high", "size_band", "beds", "doctors"]:
        if c not in ev.columns:
            ev[c] = None
    for c in ["loc_lat", "loc_lon", "loc_precision_km", "doctors", "beds", "g_rating", "g_reviews_count",
              "beds_value", "beds_low", "beds_high", "beds_p_large", "doctors_value", "doctors_low", "doctors_high",
              "evidence_score"]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev["beds_value"] = ev["beds_value"].fillna(ev["beds"])
    ev["doctors_value"] = ev["doctors_value"].fillna(ev["doctors"])
    con = load_contacts()
    ev = ev.merge(con, on="unique_id", how="left") if len(con) else ev.assign(email=None, official_website=None)
    ev["website"] = ev["website"].where(ev["website"].notna() & (ev["website"].astype(str).str.len() > 3),
                                        ev["official_website"])
    pins = query(f"SELECT pin, pin_lat, pin_lon, pin_radius_km, pin_district, pin_state FROM {DB}.pin_lookup")
    places = query(f"SELECT place, state, lat, lon, radius_km, kind, n FROM {DB}.place_lookup "
                   f"WHERE kind IN ('facility city', 'district')")
    places = places[places["kind"].isin(["facility city", "district"])]
    report = query(f"SELECT * FROM {DB}.cleaning_report")
    return (ev, pins, places, report, optional("size_model_metrics"), optional("size_model_comparison"),
            optional("size_feature_importance"))


ev, pins, places, report, size_metrics, size_comp, size_imp = load_all()

BED_CUT = 100                                   # the bed-group split used by Notebook 4 (read from its metrics)
for _, _m in size_metrics.iterrows():
    try:
        if _m["target"] == "beds" and float(_m["bed_cut"]) == float(_m["bed_cut"]):
            BED_CUT = int(_m["bed_cut"])
    except Exception:
        pass
SERVICES = sorted(ev["service"].unique())

# ---------------------------------------------------------------- languages (UI text + chat replies)
LANGS = {"English": "English", "हिन्दी": "Hindi", "বাংলা": "Bengali", "मराठी": "Marathi", "తెలుగు": "Telugu",
         "தமிழ்": "Tamil", "ગુજરાતી": "Gujarati", "ಕನ್ನಡ": "Kannada", "മലയാളം": "Malayalam", "ਪੰਜਾਬੀ": "Punjabi",
         "ଓଡ଼ିଆ": "Odia", "اردو": "Urdu"}
EN = {
    "tagline": "The right care, and the evidence to trust it.",
    "headline": "Find care<br>you can trust.",
    "stat_fac": "facility records checked", "stat_ev": "facilities with service evidence",
    "stat_strong": "service claims with strong evidence",
    "stat_google": "matched on Google Maps", "stat_svc": "life-saving services",
    "nav_find": "Find care", "nav_ask": "Ask VeriCare", "nav_gaps": "Care gaps & policy",
    "nav_ins": "India insights", "language": "Language",
    "care_needed": "Care needed", "patient_location": "Patient location",
    "loc_placeholder": "Type a city, district or PIN… or click the map", "radius": "Radius (km)",
    "weak_leads": "Weak leads", "weak_help": "Also show facilities with only weak evidence",
    "coords": "Enter exact coordinates instead", "use_these": "Use these",
    "within": "within {km} km", "n_strong": "{n} strong", "n_some": "{n} some", "n_weak": "{n} weak", "hidden": "(hidden)",
    "conf_HIGH": "Strong evidence", "conf_MEDIUM": "Some evidence · call to confirm",
    "conf_LOW": "Weak evidence · call before travelling",
    "ev_HIGH": "Strong", "ev_MEDIUM": "Some", "ev_LOW": "Weak",
    "no_phone": "no phone listed", "no_website": "no website", "no_email": "no email listed",
    "why": "Why this facility · how we decided",
    "map_caption": "Click anywhere on the map to set the patient's location. Numbers match the list.",
    "no_results": "No facilities with evidence for this service in this radius. Try a larger radius.",
    "no_strong": "No facility within {km} km has strong evidence for this service.",
    "nearest_strong": "Nearest strong option: {name} ({km} km).",
    "ask_title": "Ask VeriCare",
    "ask_caption": "Ask in plain words, in any Indian language. Answers come ONLY from our verified data, with the "
                   "evidence quoted and every facility's phone, website and email. In an emergency, call 108.",
    "ask_input": "Ask about care near a place…",
    "sug1": "Dialysis near Patna", "sug2": "Emergency C-section near Lucknow", "sug3": "NICU near 800001",
    "sug4": "Heart attack care near Jaipur", "sug5": "Blood bank in Pune", "sug6": "MRI scan near Chennai",
    "searchline": "{service} near {place} · {n} facilities within 100 km",
    "default_loc": "(your location in Find care)", "answered": "answered in {s} s",
    "no_service": "I couldn't tell which care is needed. I can help with: {services}. Please mention one, with a city "
                  "or PIN code.",
    "no_place": "I couldn't find “{place}”. Please try the city name, a nearby district, or a 6-digit PIN code.",
    "no_place_read": "I couldn't read the place name. Please type the city in English letters or give a 6-digit PIN code.",
    "llm_down": "The language model isn't available right now. The facilities below are ranked by VeriCare's evidence "
                "score.",
    "col_facility": "Facility", "col_city": "City", "col_evidence": "Evidence", "col_size": "Size", "col_phone": "Phone",
    "col_website": "Website", "col_email": "Email",
    "w_beds": "beds", "w_doctors": "doctors", "w_likely": "likely", "grp_small": "under {n} beds", "grp_large": "{n}+ beds",
    "nl_beds": "beds not listed", "nl_doctors": "doctors not listed",
    "tip_conf": "Confirmed number", "tip_pred": "Predicted by our model", "tip_unk": "Not known",
    "foot_conf": "Real data: used as-is in the ranking.",
    "foot_pred": "Not confirmed: counts ×{x} in the ranking. Call to check.",
    "foot_unk": "Counted as an average-size facility in the ranking.",
    **{f"svc::{s}": s for s in SERVICES},
}
HI = {
    "tagline": "सही इलाज, और उस पर भरोसा करने का सबूत।",
    "headline": "भरोसेमंद इलाज,<br>सबूत के साथ।",
    "stat_fac": "अस्पताल जाँचे गए", "stat_strong": "मज़बूत सबूत वाली सेवाएँ", "stat_google": "Google Maps पर मिलान",
    "stat_svc": "जीवन-रक्षक सेवाएँ", "stat_ev": "सेवा के सबूत वाले अस्पताल",
    "nav_find": "इलाज खोजें", "nav_ask": "VeriCare से पूछें", "nav_gaps": "इलाज की कमी और नीति",
    "nav_ins": "भारत की तस्वीर", "language": "भाषा",
    "care_needed": "कौन-सा इलाज चाहिए", "patient_location": "मरीज़ कहाँ है",
    "loc_placeholder": "शहर, ज़िला या पिन कोड लिखें… या नक्शे पर क्लिक करें", "radius": "दूरी (किमी)",
    "weak_leads": "कमज़ोर विकल्प भी", "weak_help": "कमज़ोर सबूत वाले अस्पताल भी दिखाएँ",
    "coords": "सटीक निर्देशांक (lat, lon) डालें", "use_these": "इन्हें इस्तेमाल करें",
    "within": "{km} किमी के अंदर", "n_strong": "{n} मज़बूत", "n_some": "{n} कुछ", "n_weak": "{n} कमज़ोर",
    "hidden": "(छिपे हुए)",
    "conf_HIGH": "मज़बूत सबूत", "conf_MEDIUM": "कुछ सबूत · पुष्टि के लिए फ़ोन करें",
    "conf_LOW": "कमज़ोर सबूत · जाने से पहले फ़ोन करें",
    "ev_HIGH": "मज़बूत", "ev_MEDIUM": "कुछ", "ev_LOW": "कमज़ोर",
    "no_phone": "फ़ोन नंबर नहीं", "no_website": "वेबसाइट नहीं", "no_email": "ईमेल नहीं",
    "why": "यह अस्पताल क्यों · हमने कैसे तय किया",
    "map_caption": "मरीज़ की जगह चुनने के लिए नक्शे पर कहीं भी क्लिक करें। नंबर सूची से मेल खाते हैं।",
    "no_results": "इस दूरी में इस सेवा के लिए कोई अस्पताल नहीं मिला। दूरी बढ़ाकर देखें।",
    "no_strong": "{km} किमी के अंदर किसी अस्पताल में इस सेवा का मज़बूत सबूत नहीं है।",
    "nearest_strong": "सबसे नज़दीकी मज़बूत विकल्प: {name} ({km} किमी)।",
    "ask_title": "VeriCare से पूछें",
    "ask_caption": "किसी भी भारतीय भाषा में सीधे शब्दों में पूछें। जवाब सिर्फ़ हमारे जाँचे गए डेटा से आते हैं: सबूत के "
                   "साथ, और हर अस्पताल का फ़ोन, वेबसाइट और ईमेल। आपात स्थिति में 108 पर कॉल करें।",
    "ask_input": "किसी जगह के पास इलाज के बारे में पूछें…",
    "sug1": "पटना में डायलिसिस कहाँ होगा?", "sug2": "लखनऊ के पास इमरजेंसी सिजेरियन", "sug3": "800001 के पास NICU",
    "sug4": "जयपुर में हार्ट अटैक का इलाज", "sug5": "पुणे में ब्लड बैंक", "sug6": "चेन्नई के पास MRI स्कैन",
    "searchline": "{place} के पास {service} · 100 किमी में {n} अस्पताल",
    "default_loc": "(‘इलाज खोजें’ में चुनी गई जगह)", "answered": "{s} सेकंड में जवाब",
    "no_service": "मैं समझ नहीं पाया कि कौन-सा इलाज चाहिए। मैं इनमें मदद कर सकता हूँ: {services}। कृपया एक चुनें, "
                  "साथ में शहर या पिन कोड बताएँ।",
    "no_place": "मुझे “{place}” नहीं मिला। कृपया शहर का नाम, पास का ज़िला या 6 अंकों का पिन कोड लिखें।",
    "no_place_read": "मैं जगह का नाम पढ़ नहीं पाया। कृपया शहर का नाम अंग्रेज़ी अक्षरों में लिखें या 6 अंकों का पिन कोड दें।",
    "llm_down": "भाषा मॉडल अभी उपलब्ध नहीं है। नीचे के अस्पताल VeriCare के सबूत-स्कोर के हिसाब से क्रम में हैं।",
    "col_facility": "अस्पताल", "col_city": "शहर", "col_evidence": "सबूत", "col_size": "आकार", "col_phone": "फ़ोन",
    "col_website": "वेबसाइट", "col_email": "ईमेल",
    "w_beds": "बेड", "w_doctors": "डॉक्टर", "w_likely": "शायद", "grp_small": "{n} से कम बेड", "grp_large": "{n}+ बेड",
    "nl_beds": "बेड की जानकारी नहीं", "nl_doctors": "डॉक्टरों की जानकारी नहीं",
    "tip_conf": "पक्की संख्या", "tip_pred": "हमारे मॉडल का अनुमान", "tip_unk": "जानकारी नहीं",
    "foot_conf": "असली डेटा: रैंकिंग में जैसा है वैसा इस्तेमाल होता है।",
    "foot_pred": "पक्का नहीं: रैंकिंग में ×{x} गिना जाता है। जाने से पहले फ़ोन करें।",
    "foot_unk": "रैंकिंग में इसे औसत आकार का अस्पताल माना गया है।",
    "svc::Dialysis": "डायलिसिस", "svc::Emergency C-section": "इमरजेंसी सिजेरियन (C-section)",
    "svc::NICU (newborn intensive care)": "NICU (नवजात शिशु गहन देखभाल)", "svc::ICU (intensive care)": "ICU (गहन चिकित्सा)",
    "svc::Cardiac surgery": "दिल की सर्जरी", "svc::Angioplasty / Cath lab": "एंजियोप्लास्टी / कैथ लैब",
    "svc::MRI scan": "MRI स्कैन", "svc::CT scan": "CT स्कैन", "svc::Burn care": "जलने का इलाज",
    "svc::Trauma / Accident care": "ट्रॉमा / दुर्घटना का इलाज", "svc::Blood bank": "ब्लड बैंक",
    "svc::Cancer care (chemo / radiation)": "कैंसर का इलाज (कीमो / रेडिएशन)",
}
try:
    from translations import TRANSLATIONS    # built-in UI text for 10 more languages (switching is instant)
except Exception:                            # missing or damaged file: the app still runs in English and Hindi
    TRANSLATIONS = {}

BUILTIN = {"Hindi": HI, **TRANSLATIONS}
PH = re.compile(r"\{\w+\}")
# keep only translations whose {placeholders} match the English text, so a typo can never crash a screen
BUILTIN = {lang: {k: v for k, v in d.items() if k in EN and sorted(PH.findall(v)) == sorted(PH.findall(EN[k]))}
           for lang, d in BUILTIN.items()}


def lang_name():
    return LANGS.get(st.session_state.get("lang", "English"), "English")


def ui():
    language = lang_name()
    return {**EN, **BUILTIN[language]} if language in BUILTIN else EN


def T(key, **kw):
    s = ui().get(key, EN.get(key, key))
    try:
        return s.format(**kw) if kw else s
    except (KeyError, IndexError, ValueError):
        return EN.get(key, key).format(**kw) if kw else EN.get(key, key)


def svc_label(s):
    return T(f"svc::{s}") if f"svc::{s}" in EN else s

# ---------------------------------------------------------------- locations
CITY_ALIASES = {
    "bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram",
    "trivandrum": "thiruvananthapuram", "cochin": "kochi", "poona": "pune", "mysore": "mysuru",
    "mangalore": "mangaluru", "belgaum": "belagavi", "hubli": "hubballi", "baroda": "vadodara",
    "allahabad": "prayagraj", "benares": "varanasi", "banaras": "varanasi", "pondicherry": "puducherry",
    "new delhi": "delhi", "calicut": "kozhikode", "trichy": "tiruchirappalli", "vizag": "visakhapatnam",
    "shimoga": "shivamogga", "gulbarga": "kalaburagi", "bellary": "ballari", "tumkur": "tumakuru",
    "chhatrapati sambhajinagar": "aurangabad", "sambhajinagar": "aurangabad", "dharashiv": "osmanabad",
    "ahilyanagar": "ahmednagar", "narmadapuram": "hoshangabad", "faizabad": "ayodhya",
}


@st.cache_resource
def location_index():
    """label -> {lat, lon, desc, note}; plus fast lookups by PIN and by place name (for the chat)."""
    p = places.copy()
    p["prio"] = p["kind"].map({"facility city": 0, "district": 1})
    p = p.sort_values(["prio", "n"], ascending=[True, False]).drop_duplicates(["place", "state"])
    opts, by_pin, by_name = {}, {}, {}
    for r in p.itertuples():
        kind = "city" if r.kind == "facility city" else "district"
        label = f"{r.place.title()}, {r.state}  ({kind})"
        opts[label] = {"lat": r.lat, "lon": r.lon, "desc": f"{r.place.title()}, {r.state}",
                       "note": f"{kind.title()} centre, approx. ±{r.radius_km:.0f} km"}
        by_name.setdefault(str(r.place).lower(), label)           # first = city with most facilities
    for alias, target in CITY_ALIASES.items():
        for r in p[p["place"] == target].itertuples():
            label = f"{alias.title()} = {r.place.title()}, {r.state}"
            opts[label] = {"lat": r.lat, "lon": r.lon, "desc": f"{alias.title()}, {r.state}",
                           "note": f"Also known as {r.place.title()}"}
            by_name.setdefault(alias, label)
    for r in pins.itertuples():
        label = f"{r.pin}  ·  PIN, {r.pin_district}, {r.pin_state}"
        opts[label] = {"lat": r.pin_lat, "lon": r.pin_lon, "desc": f"PIN {r.pin} ({r.pin_district})",
                       "note": f"PIN area centre, approx. ±{r.pin_radius_km:.0f} km"}
        by_pin[str(r.pin)] = label
    return opts, sorted(opts), by_pin, by_name


LOC_OPTS, LOC_LABELS, PIN_INDEX, PLACE_INDEX = location_index()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- ranking (same as Notebook 2)
CONF_MULT = {"HIGH": 1.0, "MEDIUM": 0.85, "LOW": 0.6}
CONF_TEXT = {"HIGH": "Strong evidence", "MEDIUM": "Some evidence · call to confirm",
             "LOW": "Weak evidence · call before travelling"}
CONF_SHORT = {"HIGH": "Strong", "MEDIUM": "Some", "LOW": "Weak"}
CONF_COLOR = {"HIGH": GREEN, "MEDIUM": AMBER, "LOW": RED}
FIELD_LABEL = {"procedure": "Procedures", "equipment": "Equipment", "capability": "Capabilities",
               "description": "Description", "google review": "Google review", "specialty": "Specialty",
               "support": "24/7"}
W_EVIDENCE, W_SIZE, W_CLOSE = 0.50, 0.33, 0.17
PREDICTED_DISCOUNT = 0.8          # a predicted bed count counts 80% as much as a listed one


def size_score(d):
    """Real bed count -> log scale (300+ beds = 1.0). Otherwise the classifier's probability of a large
    facility, discounted (x0.8) because it is a prediction."""
    s = (np.log1p(d["beds_value"]) / np.log1p(300)).clip(0, 1)
    s = np.where(d["beds_value"].isna() & d["beds_p_large"].notna(), PREDICTED_DISCOUNT * d["beds_p_large"], s)
    return pd.Series(s, index=d.index).fillna(0.3)


def rank(service, ulat, ulon, max_km):
    d = ev[ev["service"] == service].copy()
    d["distance_km"] = haversine_km(ulat, ulon, d["loc_lat"].values, d["loc_lon"].values)
    d = d[d["distance_km"] <= max_km].copy()
    d["evidence_strength"] = (d["evidence_score"] / 8).clip(upper=1.0)
    d["size_score"] = size_score(d)
    d["closeness"] = 1 / (1 + d["distance_km"] / 25)
    d["rank_score"] = (W_EVIDENCE * d["evidence_strength"] + W_SIZE * d["size_score"]
                       + W_CLOSE * d["closeness"]) * d["confidence"].map(CONF_MULT)
    return d.sort_values("rank_score", ascending=False)


CONFIRMED_TIP = {"listed": "Confirmed: listed in the Virtue Foundation dataset",
                 "stated in text": "Confirmed: stated in the facility's own description",
                 "official website": "Confirmed: published on the hospital's own website"}


def isnum(v):
    try:
        return v is not None and not pd.isna(v) and isinstance(float(v), float)
    except (TypeError, ValueError):
        return False


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def size_parts(r, what):
    """(label shown on the card, explanation shown behind the ⓘ). Real numbers as-is; otherwise a range."""
    v, src = r.get(f"{what}_value"), r.get(f"{what}_source")
    lo, hi = r.get(f"{what}_low"), r.get(f"{what}_high")
    if isnum(v) and src != "predicted":
        return f"{int(v)} {what}", CONFIRMED_TIP.get(src, "Confirmed")
    p = r.get("beds_p_large")
    group_tip = ""
    if what == "beds" and isnum(p):
        group_tip = (f" The model is {max(p, 1 - p):.0%} sure this facility has "
                     f"{'at least' if p >= 0.5 else 'fewer than'} {BED_CUT} beds.")
    # every unknown gets the model's estimate, always labelled as predicted (with how sure it is)
    if isnum(lo) and isnum(hi) and lo >= 1 and hi / lo <= 3:
        return (f"~{int(lo)}–{int(hi)} {what}",
                "Predicted by our model, not confirmed. The real number falls inside ranges like this about 80% "
                "of the time." + group_tip)
    if what == "beds" and str(r.get("beds_class_shown")).lower() == "true" and r.get("beds_class"):
        return (f"{r['beds_class']}", "Predicted by our model, not confirmed. An exact range would be too wide to be "
                "useful, so we show the size group." + group_tip)
    if what == "beds" and isnum(p):
        grp = f"{BED_CUT}+ beds" if p >= 0.5 else f"under {BED_CUT} beds"
        return (f"likely {grp}", f"Predicted by our model, not confirmed. A low-confidence estimate: the model is only "
                f"{max(p, 1 - p):.0%} sure this facility has {'at least' if p >= 0.5 else 'fewer than'} {BED_CUT} beds.")
    if isnum(lo) and isnum(hi) and hi >= 1:
        return (f"~{max(int(lo), 1)}–{int(hi)} {what}",
                "Predicted by our model, not confirmed. A wide range: the model has little to go on for this "
                "facility. The real number falls inside ranges like this about 80% of the time.")
    return f"{what} not listed", "No number in the dataset, the description or the hospital's website, and not enough " \
                                 "information to estimate one."


def loc_size(label):
    """Size label in the chosen language: '200 beds', '~6–45 doctors', '50+ beds', 'likely under 50 beds' ..."""
    if lang_name() == "English":
        return label
    if label.endswith("not listed"):
        return T(f"nl_{label.split()[0]}")
    likely = label.startswith("likely ")
    core = label[len("likely "):] if likely else label
    num = re.search(r"\d+", core)
    if core.startswith("under ") and num:
        out = T("grp_small", n=num.group())
    elif re.match(r"\d+\+ ", core):
        out = T("grp_large", n=num.group())
    else:
        out = core.replace(" beds", " " + T("w_beds")).replace(" doctors", " " + T("w_doctors"))
    return f"{T('w_likely')} {out}" if likely else out


def size_html(r, what):
    """Label + an ⓘ that opens a floating info box on hover / tap."""
    label, tip = size_parts(r, what)
    if tip.startswith("Confirmed"):
        body = tip.split(":", 1)[-1].strip()
        head, body = T("tip_conf"), body[:1].upper() + body[1:]
        foot = T("foot_conf")
    elif label.endswith("not listed"):
        head, body, foot = T("tip_unk"), tip, T("foot_unk")
    else:
        head, body = T("tip_pred"), tip.replace("Predicted by our model, not confirmed. ", "")
        foot = T("foot_pred", x=PREDICTED_DISCOUNT)
    return (f'{esc(loc_size(label))}<span class="info" tabindex="0">i<span class="tip"><b>{esc(head)}</b>{esc(body)}'
            f'<i>{esc(foot)}</i></span></span>')


def size_text(r, what):
    label, tip = size_parts(r, what)
    return f"{label} ({'confirmed' if tip.startswith('Confirmed') else 'predicted'})" \
        if not label.endswith("not listed") else label


def fmt_phone(p):
    s = re.sub(r"\.0$", "", str(p or "").strip())
    if not s or s.lower() in ("nan", "none"):
        return None
    d = re.sub(r"\D", "", s)
    if len(d) == 12 and d.startswith("91"):
        return f"+91 {d[2:7]} {d[7:]}"
    return s


def web_url(w):
    w = str(w or "").strip()
    if not w or w.lower() in ("nan", "none") or "." not in w:
        return None
    return w if w.lower().startswith("http") else "https://" + w


def first_email(e):
    e = str(e or "").split(",")[0].strip()
    return f"mailto:{e}" if "@" in e else None


def contact_table(top):
    """The table shown with every result: who, how far, how sure, and every way to contact them."""
    return pd.DataFrame({
        "#": range(1, len(top) + 1),
        T("col_facility"): top["name"].values,
        T("col_city"): top["city"].values,
        "km": top["distance_km"].round(1).values,
        T("col_evidence"): [T(f"ev_{c}") for c in top["confidence"]],
        T("col_size"): [loc_size(size_parts(r, "beds")[0]) for _, r in top.iterrows()],
        T("col_phone"): [fmt_phone(p) or "" for p in top["phone"]],
        T("col_website"): [web_url(w) or "" for w in top["website"]],
        T("col_email"): [first_email(e) or "" for e in top["email"]],
    })


def show_contacts(df, height=None):
    st.dataframe(df, hide_index=True, width="stretch", height=height or "auto", column_config={
        "#": st.column_config.NumberColumn("#", width="small"),
        "km": st.column_config.NumberColumn("km", format="%.1f", width="small"),
        **{c: st.column_config.LinkColumn(c, display_text=r"https?://(?:www\.)?([^/]+)")
           for c in df.columns if c in (EN["col_website"], T("col_website"))},
        **{c: st.column_config.LinkColumn(c, display_text=r"mailto:(.*)")
           for c in df.columns if c in (EN["col_email"], T("col_email"))},
    })


# ---------------------------------------------------------------- result card
def card(i, r):
    conf = r["confidence"]
    approx = f" ±{r['loc_precision_km']:.0f}" if isnum(r["loc_precision_km"]) and r["loc_precision_km"] > 2 else ""
    pills = f'<span class="pill pill-{conf}">{T("conf_" + conf)}</span>'
    pills += f'<span class="pill pill-grey">{icon("pin", 13)}{r["distance_km"]:.1f}{approx} km</span>'
    checks = [c for c in (r.get("checks") or []) if isinstance(c, (list, tuple)) and len(c) == 2 and c[1] is not None]
    if r.get("google_match") in ("strong", "weak") and checks:
        pills += (f'<span class="pill pill-google">Google Maps ✓ {sum(1 for c in checks if c[1])}/{len(checks)}'
                  + (f' · ★ {r["g_rating"]:.1f}' if isnum(r.get("g_rating")) else "") + "</span>")
    if len(r["warnings"]):
        pills += f'<span class="pill pill-grey">{icon("alert", 13)}{len(r["warnings"])} note{"s" if len(r["warnings"]) > 1 else ""}</span>'
    quotes = "".join(f'<div class="quote"><b>{FIELD_LABEL.get(f, f)}:</b> “{esc(t)}”</div>'
                     for f, t in list(r["quotes"])[:2])
    phone, url, mail = fmt_phone(r["phone"]), web_url(r["website"]), first_email(r.get("email"))
    contact = (f'<span class="c">{icon("phone")}{esc(phone)}</span>' if phone
               else f'<span class="c">{icon("phone")}{T("no_phone")}</span>')
    contact += (f'<a href="{esc(url)}" target="_blank">{icon("globe")}{esc(re.sub(r"^https?://(www[.])?", "", url).rstrip("/"))}</a>'
                if url else f'<span class="c">{icon("globe")}{T("no_website")}</span>')
    contact += (f'<a href="{esc(mail)}">{icon("mail")}{esc(mail[7:]).replace("@", "<span>@</span>")}</a>'
                if mail else f'<span class="c">{icon("mail")}{T("no_email")}</span>')   # split "@" stops markdown auto-linking
    contact += (f'<span class="c">{icon("bed")}{size_html(r, "beds")}</span>'
                f'<span class="c">{icon("user")}{size_html(r, "doctors")}</span>')
    with st.container(border=True):
        st.markdown(
            f'<span class="rank" style="background:{CONF_COLOR[conf]}">{i}</span>'
            f'<span class="fname">{esc(r["name"])}</span>'
            f'<div class="addr">{esc(r["address"] + ", " if r.get("address") else "")}{esc(r["city"])}, '
            f'{esc(r["state"])} {esc(r["pin"] or "")}</div>{pills}<div style="height:4px"></div>{quotes}'
            f'<div class="contact">{contact}</div>', unsafe_allow_html=True)
        with st.expander(T("why")):
            st.markdown("**Evidence quoted from the facility record**")
            for f, t in r["quotes"]:
                st.markdown(f"- *{FIELD_LABEL.get(f, f)}:* “{t}”")
            if r.get("specialty_match"):
                st.markdown(f"- *Specialty:* {r['specialty_match']}")
            if r.get("support_quote"):
                st.markdown(f"- *24/7 / emergency:* “{r['support_quote']}”")
            if checks:
                st.markdown("**Independent check (Google Maps)**  " +
                            (f"[open listing]({r['g_maps_url']})" if r.get("g_maps_url") else ""))
                st.markdown(" · ".join(("✓ " if ok else "✗ ") + label for label, ok in checks))
                if len(r.get("aliases") or []):
                    st.caption("Also known as: " + ", ".join(r["aliases"]))
            elif r.get("google_match") == "not found":
                st.caption("Could not be matched to a Google Maps listing")
            if len(r["warnings"]):
                st.markdown("**Things to be careful about**")
                for w in r["warnings"]:
                    st.markdown(f"- {w}")
            st.markdown("**How the score was calculated**")
            st.markdown(
                f"- **Service evidence (50%):** {r['score_breakdown']} → strength **{r['evidence_strength']:.2f}**\n"
                f"- **Size (33%):** {size_text(r, 'beds')} → **{r['size_score']:.2f}** "
                f"(300+ beds = 1.0; predicted sizes count ×{PREDICTED_DISCOUNT})\n"
                f"- **Closeness (17%):** 1 ÷ (1 + {r['distance_km']:.1f} ÷ 25) = **{r['closeness']:.2f}**\n"
                f"- Confidence {conf} × {CONF_MULT[conf]}"
                + (f" (lowered by: {r['confidence_lowered_by']})" if r.get("confidence_lowered_by") else "") + "\n"
                f"- **Rank score** = (0.50 × {r['evidence_strength']:.2f} + 0.33 × {r['size_score']:.2f} + "
                f"0.17 × {r['closeness']:.2f}) × {CONF_MULT[conf]} = **{r['rank_score']:.3f}**\n"
                f"- Location: {r['loc_note']}")
            if len(r["source_urls"]):
                st.markdown("**Sources:** " + " · ".join(f"[{k + 1}]({u})" for k, u in enumerate(r["source_urls"][:5])))
            st.caption(f"Record id {r['unique_id']}")


# ---------------------------------------------------------------- state: where is the patient?
if "loc" not in st.session_state:
    default = next((k for k in LOC_LABELS if k.startswith("Patna, Bihar")), LOC_LABELS[0])
    st.session_state.loc = LOC_OPTS[default]

mv = st.session_state.get("map")
if mv and mv.get("last_clicked"):
    lc = mv["last_clicked"]
    click = (round(lc["lat"], 5), round(lc["lng"], 5))
    if click != st.session_state.get("applied_click"):
        st.session_state.applied_click = click
        st.session_state.loc = {"lat": click[0], "lon": click[1],
                                "desc": f"Pinned spot ({click[0]:.4f}, {click[1]:.4f})", "note": "Dropped on the map"}


# ---------------------------------------------------------------- charts: click a bar -> details in a side panel
def people(x):
    if not isnum(x):
        return "–"
    return f"{x / 1e7:.1f} crore" if x >= 1e7 else (f"{x / 1e5:.1f} lakh" if x >= 1e5 else f"{x:,.0f}")


def pct(x, d=0):
    return f"{float(x):.{d}%}" if isnum(x) else "–"


def short(s):
    return str(s).split(" (")[0]


def style_fig(fig, height=380, legend=True):
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", size=12.5, color="#5b4a44"), height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=6, r=24, t=34 if legend else 12, b=8),
        hoverlabel=dict(bgcolor="#1c1210", bordercolor="#1c1210", font=dict(color="white", family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title=None, font=dict(size=12)),
        showlegend=legend, dragmode=False, bargap=0.28, clickmode="event+select")
    fig.update_xaxes(gridcolor="#eef2f7", zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False, linecolor=LINE, automargin=True)
    return fig


def hbar(labels, values, ids, color=BRAND, text=None, hover=None, height=None, xtitle=None, name=None):
    labels = list(labels)
    fig = go.Figure(go.Bar(y=labels, x=list(values), orientation="h", customdata=list(ids), marker=dict(
        color=color, line=dict(width=0), cornerradius=6), text=text, textposition="outside", cliponaxis=False,
        hovertemplate=hover or "%{y}: %{x}<extra></extra>", name=name or ""))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=xtitle)
    return style_fig(fig, height or max(260, 30 * len(labels) + 60), legend=False)


def picked(event):
    try:
        pts = event["selection"]["points"]
    except Exception:
        return None
    if not pts:
        return None
    cd = pts[0].get("customdata")
    if isinstance(cd, (list, tuple)):
        cd = cd[0] if cd else None
    return cd


def chart_panel(fig, key, on_pick, hint="Click any bar to see the numbers behind it", ratio=(5, 4)):
    left, right = st.columns(list(ratio), gap="large")
    with left:
        event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key=key,
                                config={"displayModeBar": False})
    pid = picked(event)
    with right:
        if pid is None:
            st.markdown(f'<div class="hint">{icon("pointer")}{esc(hint)}</div>', unsafe_allow_html=True)
        else:
            with st.container(border=True):
                on_pick(pid)
    return pid


def panel(title, sub=None, kv=None, table=None):
    st.markdown(f'<div class="ptitle">{esc(title)}</div>' + (f'<div class="psub">{esc(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)
    if kv:
        st.markdown('<div class="kv">' + "".join(f'<div class="r"><span>{esc(k)}</span><b>{esc(v)}</b></div>'
                                                  for k, v in kv.items() if not str(k).startswith("_")) + "</div>",
                    unsafe_allow_html=True)
    if table is not None and len(table):
        st.dataframe(table, hide_index=True, width="stretch")


def kpis(items):
    """items: (label, value, color). value: float share 0-1 (animated count-up) or text."""
    html = ""
    for i, (label, v, col) in enumerate(items):
        if isinstance(v, float) and isnum(v) and 0 <= v <= 1:
            val = f'<span class="count" style="--to:{round(v * 100)}"></span>'
        else:
            val = esc(v)
        html += (f'<div class="kpi" style="animation-delay:{i * .08:.2f}s"><div class="v" style="color:{col}">{val}</div>'
                 f'<div class="l">{esc(label)}</div></div>')
    st.markdown(f'<div class="kpis">{html}</div>', unsafe_allow_html=True)


def section(title, caption=None):
    st.markdown(f'<div class="section">{esc(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


# ---------------------------------------------------------------- care gaps & policy view - Notebook 6
@st.cache_resource(ttl=3600, show_spinner="Loading care-gap analysis…")
def load_gaps():
    try:
        summ = query(f"SELECT * FROM {DB}.care_gaps_summary")
        dist = query(f"SELECT * FROM {DB}.care_gaps_district")
        pol = query(f"SELECT * FROM {DB}.care_gap_policies ORDER BY rank")
    except Exception:
        return None
    for c in summ.columns:
        if c != "service":
            summ[c] = pd.to_numeric(summ[c], errors="coerce")
    for c in dist.columns:
        if c not in ("service", "pin_state", "pin_district", "centre_nearest_facility", "need_indicator"):
            dist[c] = pd.to_numeric(dist[c], errors="coerce")
    pol["rank"] = pd.to_numeric(pol["rank"], errors="coerce")
    dist["id"] = dist["pin_state"].astype(str) + "|" + dist["pin_district"].astype(str)
    return summ, dist, pol.sort_values("rank")


def share_color(s):
    s = 0 if not isnum(s) else max(0.0, min(1.0, float(s)))
    return f"#{int(30 + 200 * s):02x}{int(160 - 120 * s):02x}{int(80 - 40 * s):02x}"


def km(v):
    return f"{float(v):.0f} km" if isnum(v) else "–"


def district_panel(dist, svc, did):
    r = dist[(dist["service"] == svc) & (dist["id"] == did)]
    if not len(r):
        st.caption("No data for this district.")
        return
    r = r.iloc[0]
    kv = {"Service": svc, "People in district": people(r.get("people")),
          f"Verified within {int(r['reach_km'])} km": pct(1 - r["share_beyond"]) if isnum(r.get("share_beyond")) else "–",
          "Only unverified claims": pct(r.get("share_verify")), "Nothing claimed in reach": pct(r.get("share_nothing")),
          "People with nothing in reach": people(r.get("people_nothing")),
          "Nearest verified facility": f"{r.get('centre_nearest_facility') or '–'}",
          "Distance to it": km(r.get("centre_nearest_km")),
          "Nearest facility that claims it": km(r.get("nearest_claim_km"))}
    ni = r.get("need_indicator")
    if ni and not str(ni).startswith("none") and isnum(r.get("need")):
        kv[f"Local need (NFHS-5: {ni})"] = f"{r['need']:.2f}"
    others = dist[dist["id"] == did][["service", "share_beyond", "share_nothing"]].copy()
    others["Verified in reach"] = (1 - others["share_beyond"]).map(pct)
    others["Nothing claimed"] = others["share_nothing"].map(pct)
    panel(f"{r['pin_district']}, {r['pin_state']}", "District detail", kv,
          others.rename(columns={"service": "All services"})[["All services", "Verified in reach", "Nothing claimed"]])


def district_bar(dist, svc, by="priority", n=12, value="people_nothing", key="x", xtitle="People with nothing in reach"):
    d = dist[(dist["service"] == svc) & (dist[by] > 0)].sort_values(by, ascending=False).head(n)
    if not len(d):
        st.caption("No districts to show.")
        return
    fig = hbar([f"{a}, {b}" for a, b in zip(d["pin_district"], d["pin_state"])], d[value], d["id"].tolist(),
               color=[share_color(s) for s in d["share_nothing"]], text=[people(v) for v in d[value]],
               hover="%{y}<br>%{text}<extra></extra>", xtitle=xtitle)
    chart_panel(fig, key, lambda did: district_panel(dist, svc, did), "Click a district to see its details")


def render_gaps():
    g = load_gaps()
    if g is None:
        st.info("The care-gap analysis hasn't been generated yet: run Notebook 6 in Databricks.")
        return
    summ, dist, pol = g
    section("Where do people live far from care?",
            "For every PIN-code area we find the nearest facility for each service. Verified = strong evidence. "
            "Only unverified claims = a facility nearby claims it but the evidence is weak (verify before building). "
            "Nothing claimed = no facility within reach even claims it. ~10,000 facilities, not every facility in "
            "India: treat gaps as places to look first. People from Census 2011 district populations.")

    s2 = summ.sort_values("people_share_within_reach", ascending=False)
    fig = go.Figure()
    for col, name, color in [("people_share_within_reach", "Verified in reach", GREEN),
                             ("people_share_verification_gap", "Only unverified claims", AMBER),
                             ("people_share_possible_care_gap", "Nothing claimed in reach", RED)]:
        fig.add_bar(y=list(s2["service"]), x=list(s2[col] * 100), orientation="h", name=name,
                    marker=dict(color=color, cornerradius=4), customdata=list(s2["service"]),
                    hovertemplate="%{y}<br>" + name + ": %{x:.0f}%<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="% of India's people", range=[0, 100], ticksuffix="%")
    style_fig(fig, 470)

    def service_panel(svc):
        r = summ[summ["service"] == svc].iloc[0]
        t = dist[dist["service"] == svc].sort_values("priority", ascending=False).head(5)
        panel(svc, f"Reach: {int(r['reach_km'])} km", {
            "Verified in reach": pct(r["people_share_within_reach"]),
            "Only unverified claims": pct(r["people_share_verification_gap"]),
            "Nothing claimed in reach": pct(r["people_share_possible_care_gap"]),
            "People with nothing in reach": people(r["people_possible_care_gap"]),
            "Facilities claiming it": f"{int(r['claiming_facilities']):,}" if isnum(r.get("claiming_facilities")) else "–",
            "…with strong evidence": f"{int(r['verified_facilities']):,}" if isnum(r.get("verified_facilities")) else "–",
            "Districts with nothing in reach": f"{int(r['districts_with_nothing_claimed_within_reach']):,}"
            if isnum(r.get("districts_with_nothing_claimed_within_reach")) else "–"},
            pd.DataFrame({"Top-priority districts": t["pin_district"] + ", " + t["pin_state"],
                          "People with nothing": t["people_nothing"].map(people)}))

    svc_click = chart_panel(fig, "cov_chart", service_panel, "Click a service to see its numbers and districts")
    if svc_click and svc_click != st.session_state.get("cov_applied"):
        st.session_state.cov_applied = svc_click
        st.session_state.gap_service = svc_click

    section("Look at one service")
    svc = st.selectbox("Service", summ.sort_values("people_share_possible_care_gap", ascending=False)["service"].tolist(),
                       key="gap_service", label_visibility="collapsed")
    s = summ[summ["service"] == svc].iloc[0]
    d = dist[dist["service"] == svc]
    kpis([(f"Verified within {int(s['reach_km'])} km", float(s["people_share_within_reach"]), GREEN),
          ("Only unverified claims", float(s["people_share_verification_gap"]), AMBER),
          ("Nothing claimed in reach", float(s["people_share_possible_care_gap"]), RED),
          ("People with nothing in reach", people(s["people_possible_care_gap"]), INK)])

    view_by = st.radio("Rank districts by", ["Where to add capacity", "Where to verify first"], horizontal=True,
                       key="gap_rank", label_visibility="collapsed")
    if view_by == "Where to add capacity":
        district_bar(dist, svc, "priority", 12, "people_nothing", "gap_rank_chart")
    else:
        district_bar(dist, svc, "verify_priority", 12, "people_verify", "gap_verify_chart",
                     "People with only unverified claims")
    ni = d["need_indicator"].dropna().iloc[0] if "need_indicator" in d and d["need_indicator"].notna().any() else None
    if ni and not str(ni).startswith("none"):
        st.caption(f"Priority = people with nothing in reach × local need (NFHS-5: “{ni}”).")

    section(f"Map: where {svc.lower()} is missing")
    with st.container():
        gmap = folium.Map(location=[22.5, 80.0], zoom_start=5, tiles="OpenStreetMap")
        mx = max(d["people_nothing"].max(), 1)
        for r in d.dropna(subset=["lat", "lon"]).itertuples():
            folium.CircleMarker(
                [r.lat, r.lon], radius=3 + 14 * math.sqrt(max(r.people_nothing, 0) / mx),
                color=share_color(r.share_nothing), fill=True, fill_opacity=0.7, weight=1,
                tooltip=(f"{r.pin_district}, {r.pin_state}: nothing claimed within {int(r.reach_km)} km for "
                         f"{r.share_nothing:.0%} ({people(r.people_nothing)} people)")).add_to(gmap)
        st_folium(gmap, height=520, use_container_width=True, returned_objects=[], key="gap_map")
        st.caption("Each circle is a district: size = people with nothing in reach, colour = share with nothing "
                   "(green = covered, red = possible care gap).")

    section("Policy recommendations", "Each recommendation comes with the data behind it. Click the chart to dig in.")
    for p in pol.itertuples():
        with st.container(border=True):
            st.markdown(f'<span class="ins-n">{int(p.rank)}</span><span class="ins-h">{esc(p.title)}</span> '
                        f'&nbsp;<span class="pill pill-grey">{esc(p.theme)}</span>'
                        f'<div class="ins-d">{esc(p.detail)}</div>', unsafe_allow_html=True)
            policy_chart(p, summ, dist)
            st.caption(f"Evidence: {p.evidence}")


HIGH_STAKES = ["Emergency C-section", "NICU (newborn intensive care)", "Trauma / Accident care", "Dialysis"]


def policy_chart(p, summ, dist):
    key = f"pol_{int(p.rank)}"
    m = re.search(r"service = (.+)$", str(p.evidence))
    svc = m.group(1).strip() if m else None
    theme = str(p.theme)
    if theme == "Verify first":
        s = summ[summ["service"].isin(HIGH_STAKES)]
        fig = go.Figure()
        fig.add_bar(y=list(s["service"].map(short)), x=list(s["claiming_facilities"]), orientation="h",
                    name="Claim the service", marker=dict(color="#ffd2c4", cornerradius=4), customdata=list(s["service"]),
                    hovertemplate="%{y}: %{x:,} claim it<extra></extra>")
        fig.add_bar(y=list(s["service"].map(short)), x=list(s["verified_facilities"]), orientation="h",
                    name="Strong evidence", marker=dict(color=BRAND, cornerradius=4), customdata=list(s["service"]),
                    hovertemplate="%{y}: %{x:,} with strong evidence<extra></extra>")
        fig.update_layout(barmode="group")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Facilities")
        style_fig(fig, 300)

        def pick(sv):
            r = summ[summ["service"] == sv].iloc[0]
            t = dist[dist["service"] == sv].sort_values("verify_priority", ascending=False).head(6)
            panel(sv, "Claims vs strong evidence", {
                "Facilities claiming it": f"{int(r['claiming_facilities']):,}",
                "With strong evidence": f"{int(r['verified_facilities']):,}",
                "Share verified": pct(r["verified_facilities"] / max(r["claiming_facilities"], 1)),
                "People with only unverified claims nearby": people(r["people_verification_gap"])},
                pd.DataFrame({"Verify first in": t["pin_district"] + ", " + t["pin_state"],
                              "People": t["people_verify"].map(people)}))
        chart_panel(fig, key, pick, "Click a service")
    elif theme in ("Where to add capacity", "Dialysis"):
        svc = svc if svc in set(dist["service"]) else "Dialysis"
        district_bar(dist, svc, "priority", 10, "people_nothing", key)
    elif theme == "Emergency referral":
        em = dist[dist["service"].isin(["Emergency C-section", "Trauma / Accident care"])]
        far = em[em["nearest_claim_km"] > 100].sort_values("nearest_claim_km", ascending=False) \
            .drop_duplicates("id").head(20)
        if not len(far):
            return
        fig = hbar([f"{a}, {b}" for a, b in zip(far["pin_district"], far["pin_state"])], far["nearest_claim_km"],
                   far["id"].tolist(), color=RED, text=[f"{v:.0f} km" for v in far["nearest_claim_km"]],
                   hover="%{y}<br>%{text} to the nearest claim<extra></extra>",
                   xtitle="km to the nearest facility that even claims emergency C-section or trauma care")

        def pick(did):
            rows = em[em["id"] == did]
            r0 = rows.iloc[0]
            kv = {}
            for x in rows.itertuples():
                kv[f"{short(x.service)}: nearest claim"] = km(x.nearest_claim_km)
                kv[f"{short(x.service)}: nearest verified"] = km(x.centre_nearest_km)
            kv["People in district"] = people(r0["people"])
            panel(f"{r0['pin_district']}, {r0['pin_state']}", "Emergency care distances", kv)
        chart_panel(fig, key, pick, "Click a district")
    elif theme == "Data quality":
        u = ev.drop_duplicates("unique_id")
        rows = [("No bed count stated", int(u["beds"].isna().sum()))]
        for _, r in report.iterrows():
            c = str(r["check"])
            if re.search(r"Location confidence: low|Merge risk: high|shared with 3|Doctor count of 1", c):
                n = pd.to_numeric(r["count"], errors="coerce")
                rows.append((c, int(n) if isnum(n) else 0))
        df = pd.DataFrame(rows, columns=["label", "n"]).set_index("label")
        fig = hbar(df.index, df["n"], df.index.tolist(), color=AMBER, text=[f"{v:,}" for v in df["n"]],
                   hover="%{y}: %{x:,}<extra></extra>", xtitle="Facility records")
        chart_panel(fig, key, lambda lab: panel(lab, "Data-quality check", {
            "Records": f"{int(df.loc[lab, 'n']):,}",
            "Share of searchable facilities": pct(df.loc[lab, "n"] / max(len(u), 1))}), "Click a check")
    elif theme == "Verification":
        c = ev.groupby(["service", "confidence"]).size().unstack(fill_value=0) \
            .reindex(columns=["HIGH", "MEDIUM", "LOW"], fill_value=0)
        c = c.loc[(c["HIGH"] / c.sum(axis=1)).sort_values(ascending=False).index]
        fig = go.Figure()
        for k, name in [("HIGH", "Strong"), ("MEDIUM", "Some"), ("LOW", "Weak")]:
            fig.add_bar(y=[short(x) for x in c.index], x=list(c[k]), orientation="h", name=f"{name} evidence",
                        marker=dict(color=CONF_COLOR[k], cornerradius=3), customdata=list(c.index),
                        hovertemplate="%{y}: %{x:,} " + name.lower() + "<extra></extra>")
        fig.update_layout(barmode="stack")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Facility-service claims")
        style_fig(fig, 420)

        def pick(sv):
            r = c.loc[sv]
            tot = int(r.sum())
            panel(sv, "How strong are the claims?", {
                "Claims": f"{tot:,}", "Strong": f"{int(r['HIGH']):,} ({pct(r['HIGH'] / tot)})",
                "Some": f"{int(r['MEDIUM']):,} ({pct(r['MEDIUM'] / tot)})",
                "Weak": f"{int(r['LOW']):,} ({pct(r['LOW'] / tot)})"})
        chart_panel(fig, key, pick, "Click a service")


# ---------------------------------------------------------------- India insights view - Notebook 7
def safe(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


LIFE_CRITICAL = ["Emergency C-section", "NICU (newborn intensive care)", "Trauma / Accident care",
                 "ICU (intensive care)", "Dialysis", "Blood bank"]
TEXT_COLS = ("state", "region", "pin_state", "pin_district", "band", "key", "label", "series", "info", "headline", "detail")


@st.cache_resource(ttl=3600, show_spinner="Loading India insights…")
def load_insights():
    try:
        ins = query(f"SELECT * FROM {DB}.india_insights ORDER BY n")
    except Exception:
        return None
    ins["n"] = pd.to_numeric(ins["n"], errors="coerce")
    if "ref" in ins:
        ins["ref"] = pd.to_numeric(ins["ref"], errors="coerce")
    names = {safe(s): s for s in list(SERVICES) + LIFE_CRITICAL}
    score = optional("india_state_scorecard").rename(columns=names)
    reg = optional("india_region_coverage").rename(columns=names)
    cs = optional("india_csection_districts")
    pts = optional("india_insight_points")
    for df in (score, reg, cs, pts):
        for c in df.columns:
            if c not in TEXT_COLS:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    if len(pts):
        pts["info"] = pts["info"].map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    return ins.sort_values("n"), score, reg, cs, pts


def region_of(state):
    eag = ["Bihar", "Uttar Pradesh", "Madhya Pradesh", "Rajasthan", "Jharkhand", "Odisha", "Chhattisgarh", "Uttarakhand"]
    south = ["Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana"]
    return "EAG states" if state in eag else ("South" if state in south else "Rest of India")


def points_panel(pts, key):
    def show(label):
        r = pts[(pts["key"] == key) & (pts["label"] == label)]
        if not len(r):
            st.caption("No details.")
            return
        info = r.iloc[0]["info"] or {}
        panel(label, None, info, pd.DataFrame(info["_table"]) if info.get("_table") else None)
    return show


@st.cache_resource
def district_coords():
    """'State|District' -> (lat, lon), and state -> centre, from the care-gap tables (Notebook 6)."""
    g = load_gaps()
    if g is None:
        return {}, {}
    d = g[1].dropna(subset=["lat", "lon"]).drop_duplicates("id")
    by_id = {i: (la, lo) for i, la, lo in zip(d["id"], d["lat"], d["lon"])}
    by_state = d.groupby("pin_state")[["lat", "lon"]].mean()
    return by_id, {s: (r.lat, r.lon) for s, r in by_state.iterrows()}


def label_to_id(label):
    """'Purnia, Bihar' -> 'Bihar|Purnia'"""
    parts = str(label).rsplit(", ", 1)
    return f"{parts[1]}|{parts[0]}" if len(parts) == 2 else str(label)


def map_fig(df, colorscale=None, cbar=None, cmin=None, cmax=None, height=470):
    """df columns: lat, lon, id, c (colour value or colour), s (dot size), h (hover text). Dots click like bars."""
    df = df.dropna(subset=["lat", "lon"])
    hover = df["h"]
    marker = dict(size=list(df["s"]), color=list(df["c"]), opacity=0.82)
    if colorscale:
        marker.update(colorscale=colorscale, showscale=True, cmin=cmin, cmax=cmax,
                      colorbar=dict(title=dict(text=cbar or "", side="right"), thickness=10, len=0.7))
    fig = go.Figure(go.Scattermap(lat=list(df["lat"]), lon=list(df["lon"]), mode="markers", customdata=list(df["id"]),
                                  text=list(hover), hovertemplate="%{text}<extra></extra>", marker=marker))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lat=22.8, lon=81.0), zoom=3.5),
                      margin=dict(l=0, r=0, t=0, b=0), height=height, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, system-ui, sans-serif"), clickmode="event+select",
                      hoverlabel=dict(bgcolor="#1c1210", font=dict(color="white", family="Inter")))
    return fig


def chart_or_map(ck, fig, make_map, pick, hint="Click any bar to see the numbers behind it"):
    mode = st.radio("Show as", ["Chart", "Map"], horizontal=True, key=f"{ck}_mode", label_visibility="collapsed")
    if mode.endswith("Map"):
        mf = make_map()
        if mf is not None and len(mf.data[0].lat or []):
            return chart_panel(mf, f"{ck}_map", pick, "Click a dot on the map (scroll to zoom, drag to move)")
        st.caption("No map positions available for this insight (run Notebook 6).")
    return chart_panel(fig, ck, pick, hint)


def with_coords(df, id_col="id"):
    by_id, _ = district_coords()
    df = df.copy()
    df["lat"] = [by_id.get(i, (None, None))[0] for i in df[id_col]]
    df["lon"] = [by_id.get(i, (None, None))[1] for i in df[id_col]]
    return df


def insight_chart(r, score, reg, cs, pts):
    key, ref = r.key, getattr(r, "ref", None)
    P = pts[pts["key"] == key] if len(pts) else pd.DataFrame()
    ck = f"ins_{key}"
    if key == "csection_paradox" and len(cs):
        c = cs.dropna(subset=["cs_rate", "verified_share"]).copy()
        c["region"] = c["pin_state"].map(region_of)
        c["id"] = c["pin_state"] + "|" + c["pin_district"]
        fig = go.Figure()
        fig.add_vrect(x0=10, x1=15, fillcolor="#ef4f3c", opacity=0.10, line_width=0,
                      annotation_text="WHO healthy range", annotation_position="top left")
        fig.add_vline(x=5, line_dash="dash", line_color=RED, line_width=1)
        for reg_name in ["Rest of India", "EAG states", "South"]:
            t = c[c["region"] == reg_name]
            fig.add_scatter(x=list(t["cs_rate"]), y=list(t["verified_share"] * 100), mode="markers", name=reg_name,
                            customdata=list(t["id"]), text=list(t["pin_district"] + ", " + t["pin_state"]),
                            marker=dict(color=REGION_COLOR[reg_name], size=7, opacity=0.75,
                                        line=dict(width=0.5, color="white")),
                            hovertemplate="%{text}<br>C-section rate %{x:.1f}%<br>verified care %{y:.0f}%<extra></extra>")
        fig.update_xaxes(title="District C-section rate, % of births (NFHS-5)")
        fig.update_yaxes(title="Verified C-section care ≤ 50 km (% of people)")
        style_fig(fig, 420)

        def pick(did):
            x = c[c["id"] == did].iloc[0]
            panel(f"{x['pin_district']}, {x['pin_state']}", x["region"], {
                "C-section rate (NFHS-5)": f"{x['cs_rate']:.1f}%", "Band": x["band"],
                "Verified C-section care within 50 km": pct(x["verified_share"]),
                "Nothing claimed within 50 km": pct(x["share_nothing"]), "District population": people(x["people"])})

        def mk():
            m = with_coords(c)
            m["c"], m["s"] = m["cs_rate"], 6 + 12 * (1 - m["verified_share"].clip(0, 1))
            m["h"] = [f"{d}, {s}<br>C-section rate {v:.1f}%<br>verified care {w:.0%}" for d, s, v, w in
                      zip(m["pin_district"], m["pin_state"], m["cs_rate"], m["verified_share"])]
            return map_fig(m, [[0, RED], [0.2, AMBER], [0.35, GREEN], [1, "#8e1f45"]], "C-section %", 0, 40)
        chart_or_map(ck, fig, mk, pick, "Click any district dot")
        st.caption("Map: colour = C-section rate (red = too few, green = near the WHO range, purple = too many); "
                   "bigger dot = less verified C-section care nearby.")
    elif key == "north_south" and len(reg):
        regs = ["EAG states", "Rest of India", "South"]
        rg = reg.set_index("region")
        svcs = [s for s in LIFE_CRITICAL if s in rg.columns]
        fig = go.Figure()
        for rn in regs:
            if rn in rg.index:
                fig.add_bar(y=[short(s) for s in svcs], x=[rg.loc[rn, s] * 100 for s in svcs], orientation="h", name=rn,
                            marker=dict(color=REGION_COLOR[rn], cornerradius=4), customdata=[f"{rn}|{s}" for s in svcs],
                            hovertemplate=rn + " · %{y}: %{x:.0f}%<extra></extra>")
        fig.update_layout(barmode="group")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="People with verified care within 50 km (%)", ticksuffix="%")
        style_fig(fig, 440)

        def pick(cid):
            rn, sv = cid.split("|", 1)
            x = rg.loc[rn]
            panel(f"{rn} · {short(sv)}", "Verified access within 50 km", {
                "Verified access": pct(x[sv]), "Share of India's people": pct(x["people_share"]),
                "Facilities per crore people": f"{x['facilities_per_crore']:.0f}"},
                pd.DataFrame({"Service": [short(s) for s in svcs],
                              **{k: [pct(rg.loc[k, s]) for s in svcs] for k in regs if k in rg.index}}))
        chart_panel(fig, ck, pick)
    elif key == "emergency_distance" and len(reg):
        rg = reg.set_index("region")
        cols = [(c, n) for c, n in [("typical_km_emergency_c_section", "Emergency C-section"),
                                    ("typical_km_trauma_accident_care", "Trauma care")] if c in rg.columns]
        fig = go.Figure()
        for rn in ["EAG states", "Rest of India", "South"]:
            if rn in rg.index:
                fig.add_bar(y=[n for _, n in cols], x=[rg.loc[rn, c] for c, _ in cols], orientation="h", name=rn,
                            marker=dict(color=REGION_COLOR[rn], cornerradius=4), customdata=[f"{rn}|{c}" for c, _ in cols],
                            text=[km(rg.loc[rn, c]) for c, _ in cols], textposition="outside", cliponaxis=False,
                            hovertemplate=rn + " · %{y}: %{x:.0f} km<extra></extra>")
        fig.update_layout(barmode="group")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Typical km to the nearest facility with strong evidence")
        style_fig(fig, 320)

        def pick(cid):
            rn, c = cid.split("|", 1)
            x = rg.loc[rn]
            avg = c.replace("typical", "median")
            panel(f"{rn} · {dict(cols)[c]}", "Distance to verified emergency care", {
                "Typical (median) distance": km(x[c]),
                "Average distance (people-weighted)": km(x[avg]) if avg in x else "–",
                "Share of India's people": pct(x["people_share"])})
        chart_panel(fig, ck, pick)
    elif key == "metro_concentration" and len(P):
        P = P.sort_values("value", ascending=False)
        fig = hbar(P["label"].map(short), P["value"], P["label"].tolist(), color=BRAND,
                   text=[f"{v:.0f}%" for v in P["value"]], hover="%{y}: %{x:.0f}% in the 8 metros<extra></extra>",
                   xtitle="% of facilities with strong evidence that are in the 8 metros")
        if isnum(ref):
            fig.add_vline(x=float(ref), line_dash="dash", line_color=RED,
                          annotation_text=f"metros' share of people: {float(ref):.0f}%", annotation_position="top")
        fig.update_xaxes(ticksuffix="%")
        chart_panel(fig, ck, points_panel(pts, key))
    elif key in ("insured_nowhere", "diabetes_double_burden") and len(P):
        allP = P.copy()
        P = P.sort_values("value", ascending=False).head(15)
        what = "households insured" if key == "insured_nowhere" else "adults high sugar"
        fig = hbar(P["label"], P["value"], P["label"].tolist(), color=list(P["value2"]),
                   text=[people(v) for v in P["value"]], hover="%{y}<br>%{text} with nothing in reach<extra></extra>",
                   xtitle=f"People with no {'ICU' if key == 'insured_nowhere' else 'dialysis'} even claimed within 50 km")
        fig.update_traces(marker=dict(colorscale=[[0, "#ffd2c4"], [1, "#8e1f45"]], showscale=True,
                                      colorbar=dict(title=dict(text=f"% {what}", side="right"), thickness=10)))

        def mk():
            m = allP.assign(id=allP["label"].map(label_to_id))
            m = with_coords(m).assign(id=allP["label"].values)          # click -> same panel as the bars
            mx = max(m["value"].max(), 1)
            m["c"], m["s"] = m["value2"], 7 + 18 * np.sqrt(m["value"].clip(lower=0) / mx)
            m["h"] = [f"{lab}<br>{people(v)} with nothing in reach<br>{v2:.0f}% {what}"
                      for lab, v, v2 in zip(m["label"], m["value"], m["value2"])]
            return map_fig(m, [[0, "#ffd2c4"], [1, "#8e1f45"]], f"% {what}")
        chart_or_map(ck, fig, mk, points_panel(pts, key), "Click a district")
    elif key == "aspirational" and len(P):
        allP = P.copy()
        P = P.sort_values("value", ascending=False).head(25).copy()
        P["idx"] = P["value"] / P["value"].max() * 100
        on = P["series"].str.startswith("NITI")
        fig = hbar(P["label"], P["idx"], P["label"].tolist(), color=np.where(on, BRAND2, "#cbd5e1").tolist(),
                   hover="%{y}<br>priority index %{x:.0f}<extra></extra>", xtitle="Priority index (top district = 100)")
        for name, col in [("NITI Aayog Aspirational District", BRAND2), ("Not on the list", "#cbd5e1")]:
            fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=col, symbol="square"), name=name)
        fig.update_layout(showlegend=True, margin=dict(t=34))

        def mk():
            m = with_coords(allP.assign(id=allP["label"].map(label_to_id))).assign(id=allP["label"].values)
            on_list = m["series"].str.startswith("NITI")
            m["c"] = np.where(on_list, BRAND2, "#94a3b8")
            m["s"] = 8 + 16 * m["value"] / max(m["value"].max(), 1)
            m["h"] = [f"{lab}<br>{'NITI Aayog Aspirational District' if o else 'not on the list'}"
                      for lab, o in zip(m["label"], on_list)]
            return map_fig(m)
        chart_or_map(ck, fig, mk, points_panel(pts, key), "Click a district to see which services it lacks")
        st.caption("Teal = on NITI Aayog's Aspirational Districts list; grey = not on it. Bigger = higher priority.")
    elif key == "public_invisible" and len(P):
        own, rural = P[P["series"] == "owner"], P[P["series"] == "rural"]
        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                            subplot_titles=("Who runs the facilities in web data", "Rural public centres: online vs reality"))
        fig.add_bar(y=list(own["label"]), x=list(own["value"]), orientation="h", customdata=list(own["label"]),
                    marker=dict(color=[BRAND, BRAND2][:len(own)], cornerradius=4), text=[f"{v:,.0f}" for v in own["value"]],
                    textposition="outside", cliponaxis=False, row=1, col=1, name="Facilities",
                    hovertemplate="%{y}: %{x:,}<extra></extra>", showlegend=False)
        ylab = ["PHCs" if s.startswith("Primary") else "CHCs" for s in rural["label"]]
        fig.add_bar(y=ylab, x=list(rural["value2"]), orientation="h", customdata=list(rural["label"]),
                    name="Run by the government", marker=dict(color="#cbd5e1", cornerradius=4), row=1, col=2,
                    hovertemplate="%{y}: %{x:,} run by government<extra></extra>")
        fig.add_bar(y=ylab, x=list(rural["value"]), orientation="h", customdata=list(rural["label"]),
                    name="Found online", marker=dict(color=RED, cornerradius=4), row=1, col=2,
                    hovertemplate="%{y}: %{x:,} found online<extra></extra>")
        fig.update_xaxes(type="log", row=1, col=2, title="count (log scale)")
        fig.update_xaxes(row=1, col=1, range=[0, max(own["value"].max() * 1.25, 1)])
        style_fig(fig, 320)
        fig.update_layout(barmode="overlay", showlegend=True, margin=dict(t=40, b=10),
                          legend=dict(orientation="h", yanchor="top", y=-0.22, x=0.55))
        chart_panel(fig, ck, points_panel(pts, key))
    elif key == "state_scorecard" and len(score):
        sc = score.sort_values("life_critical_access", ascending=False)
        fig = hbar(sc["state"], sc["life_critical_access"] * 100, sc["state"].tolist(),
                   color=[REGION_COLOR.get(x, GREY) for x in sc["region"]],
                   text=[pct(v) for v in sc["life_critical_access"]], hover="%{y}: %{x:.0f}%<extra></extra>",
                   xtitle="Average % of people with verified life-critical care within 50 km", height=24 * len(sc) + 80)
        fig.update_xaxes(ticksuffix="%")

        def pick(stt):
            x = sc[sc["state"] == stt].iloc[0]
            have = [s for s in LIFE_CRITICAL if s in sc.columns]
            panel(f"#{int(x['rank'])} {stt}", x["region"], {
                "Life-critical access": pct(x["life_critical_access"]), "People": people(x["people"]),
                "Facilities per crore people": f"{x['facilities_per_crore']:.0f}",
                "Data quality (location confidence)": pct(x["data_quality"])},
                pd.DataFrame({"Service": [short(s) for s in have], "Verified within 50 km": [pct(x[s]) for s in have]}))

        def mk():
            _, by_state = district_coords()
            m = sc.assign(id=sc["state"])
            m["lat"] = [by_state.get(s, (None, None))[0] for s in m["state"]]
            m["lon"] = [by_state.get(s, (None, None))[1] for s in m["state"]]
            m["c"] = m["life_critical_access"] * 100
            m["s"] = 10 + 26 * np.sqrt(m["people"] / max(m["people"].max(), 1))
            m["h"] = [f"#{int(k)} {s}<br>life-critical access {v:.0%}" for k, s, v in
                      zip(m["rank"], m["state"], m["life_critical_access"])]
            return map_fig(m, [[0, RED], [0.5, AMBER], [1, GREEN]], "access %", 0, 100)
        chart_or_map(ck, fig, mk, pick, "Click a state to see its six services")
        st.caption("Map: one bubble per state, size = population, colour = verified life-critical access "
                   "(red = low, green = high).")
    elif key in ("metro_concentration", "insured_nowhere", "diabetes_double_burden", "aspirational", "public_invisible"):
        st.caption("Re-run Notebook 7 to add the chart for this insight.")


def render_insights():
    g = load_insights()
    if g is None:
        st.info("The India insights haven't been generated yet: run Notebook 7 in Databricks.")
        return
    ins, score, reg, cs, pts = g
    section("What this data tells us about healthcare in India",
            "Built from the facility dataset, NFHS-5 district health data, Census 2011 populations and our "
            "verified-access analysis. Verified = strong evidence of the service in this dataset. Click any chart.")
    for r in ins.itertuples():
        with st.container(border=True):
            st.markdown(f'<span class="ins-n">{int(r.n)}</span><span class="ins-h">{esc(r.headline)}</span>'
                        f'<div class="ins-d">{esc(r.detail)}</div>', unsafe_allow_html=True)
            insight_chart(r, score, reg, cs, pts)


# ---------------------------------------------------------------- Ask VeriCare (LLM, grounded in our data)
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


@st.cache_resource
def workspace_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


def gemini_body(messages, max_tokens, temperature):
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    gen = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if "2.5" in GEMINI_MODEL and "pro" not in GEMINI_MODEL:
        gen["thinkingConfig"] = {"thinkingBudget": 0}            # no hidden "thinking": answers start ~instantly
    return {"systemInstruction": {"parts": [{"text": system}]}, "generationConfig": gen,
            "contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                         for m in messages if m["role"] != "system"]}


def llm(messages, max_tokens=400, temperature=0.2):
    """messages: [{'role': 'system'|'user'|'assistant', 'content': str}] -> reply text"""
    if GEMINI_KEY:
        import requests
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                          params={"key": GEMINI_KEY}, timeout=40, json=gemini_body(messages, max_tokens, temperature))
        r.raise_for_status()
        return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
    resp = workspace_client().serving_endpoints.query(
        name=LLM_ENDPOINT, max_tokens=max_tokens, temperature=temperature,
        messages=[ChatMessage(role=ChatMessageRole(m["role"]), content=m["content"]) for m in messages])
    return resp.choices[0].message.content


def llm_stream(messages, max_tokens=400, temperature=0.2):
    """Yields the answer piece by piece, so the first words appear in about a second."""
    if GEMINI_KEY:
        import requests
        with requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent",
                           params={"key": GEMINI_KEY, "alt": "sse"}, timeout=40, stream=True,
                           json=gemini_body(messages, max_tokens, temperature)) as r:
            r.raise_for_status()
            r.encoding = "utf-8"
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    j = json.loads(line[5:])
                    for part in (j.get("candidates") or [{}])[0].get("content", {}).get("parts", []):
                        if part.get("text"):
                            yield part["text"]
        return
    sent = False
    try:
        client = workspace_client().serving_endpoints.get_open_ai_client()
        for ch in client.chat.completions.create(model=LLM_ENDPOINT, messages=messages, max_tokens=max_tokens,
                                                 temperature=temperature, stream=True):
            if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content:
                sent = True
                yield ch.choices[0].delta.content
    except Exception:
        if sent:
            raise
        yield llm(messages, max_tokens, temperature)          # streaming not available: one-shot answer


SERVICE_WORDS = {
    "Dialysis": ["dialysis", "kidney", "renal", "haemodialysis", "hemodialysis", "डायलिसिस", "गुर्दा", "किडनी"],
    "Emergency C-section": ["c section", "csection", "caesarean", "cesarean", "caesarian", "cesarian",
                            "delivery", "labour pain", "labor pain", "pregnan", "सिजेरियन", "डिलीवरी", "प्रसव", "ऑपरेशन से"],
    "NICU (newborn intensive care)": ["nicu", "newborn", "neonatal", "premature", "new born", "नवजात", "शिशु"],
    "ICU (intensive care)": ["icu", "intensive care", "critical care", "ventilator", "आईसीयू"],
    "Cardiac surgery": ["cardiac surgery", "heart surgery", "bypass", "cabg", "open heart", "valve replacement"],
    "Angioplasty / Cath lab": ["angioplasty", "cath lab", "cathlab", "stent", "angiography", "heart attack",
                               "दिल का दौरा", "हार्ट अटैक"],
    "MRI scan": ["mri"],
    "CT scan": ["ct scan", "ct", "cat scan", "सीटी"],
    "Burn care": ["burn", "burns", "जलने", "झुलस"],
    "Trauma / Accident care": ["trauma", "accident", "injury", "injured", "fracture", "दुर्घटना", "एक्सीडेंट", "चोट"],
    "Blood bank": ["blood bank", "blood", "khoon", "रक्त", "खून", "ब्लड"],
    "Cancer care (chemo / radiation)": ["cancer", "chemo", "chemotherapy", "radiation", "radiotherapy", "oncology",
                                        "tumour", "tumor", "कैंसर"],
}
STOP = {"near", "care", "the", "and", "for", "where", "get", "can", "ke", "paas", "pass", "kahan", "kaha", "hoga",
        "hospital", "best", "city", "district", "need", "want", "find", "me", "my", "in", "at", "of", "to", "is",
        "a", "an", "i", "with", "around", "close", "good", "top", "mein", "hai", "kya", "help", "please", "emergency",
        "nearby", "urgent", "surgery", "scan", "bank", "lab", "heart", "blood", "baby"}


INDIC_SPLIT = re.compile(r"[^\wऀ-෿؀-ۿ]+")   # keep letters + vowel signs of Indian scripts


def parse_query(q):
    """Fast path (no language model): find the service and place in the question with our own word lists."""
    low = " " + re.sub(INDIC_SPLIT, " ", q.lower()) + " "
    svc, best = None, 0
    for s, words in SERVICE_WORDS.items():
        if s not in SERVICES:
            continue
        for w in words:
            w2 = re.sub(INDIC_SPLIT, " ", w.lower()).strip()
            hit = (f" {w2} " in low) if (w2.isascii() and len(w2) < 6) else (w2 in low)
            if hit and len(w2) > best:
                svc, best = s, len(w2)
    loc = None
    m = re.search(r"(-?\d{1,2}\.\d+)\s*,\s*(-?\d{2,3}\.\d+)", q)
    if m:
        loc = {"lat": float(m.group(1)), "lon": float(m.group(2)), "desc": f"coordinates {m.group(1)}, {m.group(2)}",
               "note": None}
    if loc is None:
        m = re.search(r"\b([1-9]\d{5})\b", q)
        if m and m.group(1) in PIN_INDEX:
            loc = LOC_OPTS[PIN_INDEX[m.group(1)]]
    if loc is None:
        toks = []
        for t in low.split():
            if not t.isascii():
                continue
            # Hinglish glues words onto place names: "puneke", "patnamein", "delhise" -> pune, patna, delhi
            if t not in PLACE_INDEX:
                for suf in ("wala", "wale", "mein", "me", "ke", "ka", "ki", "se", "ko", "mai", "pe", "par"):
                    if t.endswith(suf) and len(t) - len(suf) >= 3 and t[:-len(suf)] in PLACE_INDEX:
                        t = t[:-len(suf)]
                        break
            toks.append(t)
        for n in (3, 2, 1):
            for i in range(len(toks) - n + 1):
                ph = " ".join(toks[i:i + n])
                if len(ph) >= 3 and ph not in STOP and ph in PLACE_INDEX:
                    loc = LOC_OPTS[PLACE_INDEX[ph]]
                    break
            if loc:
                break
    return svc, loc


def resolve_place(text):
    t = str(text or "").strip()
    _, loc = parse_query(t)
    if loc:
        return loc
    low = t.lower()
    hits = [k for k in LOC_LABELS if k.lower().startswith(low)] if low else []
    if not hits and low:
        hits = [k for k in LOC_LABELS if low in k.lower()]
    if not hits:
        return None
    return LOC_OPTS[next((k for k in hits if k.endswith("(city)")), hits[0])]


EXTRACT_PROMPT = ("You extract search parameters for a healthcare facility finder in India. The user may write in "
                  "English, Hindi or another Indian language. Services available: {services}. Reply with JSON only, "
                  'no other text: {{"service": <exactly one name from the list, or null>, "location": <the place '
                  "name, PIN code or 'lat, lon' the user mentions, in English spelling, or null>}}")
ANSWER_PROMPT = (
    "You are VeriCare, an assistant for community health workers and families in India. Answer ONLY from the "
    "FACILITY DATA given to you. Never invent facilities, services, phone numbers or numbers. Recommend the best 2-3 "
    "facilities: name, distance and evidence level, and quote their evidence in double quotes. If evidence is weak, a "
    "size is predicted, or there is a caution, say so plainly. If no facility has strong evidence, say so. Contact "
    "details are shown in a table under your answer, so don't repeat them. End by reminding the user to call before "
    "travelling, and in an emergency to call 108. No medical diagnoses. Reply in {language}; if the user clearly "
    "wrote in a different language, reply in the user's language instead. Keep facility names as written. Use short "
    "bullet points. Under 130 words.")


def facility_context(top):
    lines = []
    for i, (_, r) in enumerate(top.iterrows(), 1):
        quotes = "; ".join(f'"{t}"' for _, t in list(r["quotes"])[:2]) or "(only a related specialty is listed)"
        warns = "; ".join(list(r["warnings"])[:2]) or "none"
        lines.append(f"{i}. {r['name']}, {r['city']} | {r['distance_km']:.1f} km | evidence {r['confidence']} | "
                     f"size: {size_text(r, 'beds')}\n   evidence: {quotes}\n   cautions: {warns}")
    return "\n".join(lines)


def show_answer(msg):
    if msg.get("meta"):
        st.markdown(f'<div class="searchline">{msg["meta"]}</div>', unsafe_allow_html=True)
    st.markdown(msg["content"])
    if msg.get("table") is not None:
        show_contacts(msg["table"])




def render_ask():
    st.caption(T("ask_caption"))
    engine = f"Gemini ({GEMINI_MODEL})" if GEMINI_KEY else f"Databricks model serving ({LLM_ENDPOINT})"
    st.caption(f"Language model: {engine}")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if not st.session_state.chat:
        cols = st.columns(3)
        for i, s in enumerate(T(f"sug{k}") for k in range(1, 7)):
            if cols[i % 3].button(s, key=f"sug_{i}", width="stretch"):
                st.session_state.pending_q = s
                st.rerun()
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/verified_user:"):
            if msg["role"] == "assistant":
                show_answer(msg)
            else:
                st.markdown(msg["content"])
    q = st.chat_input(T("ask_input")) or st.session_state.pop("pending_q", None)
    if not q:
        return
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(q)
    with st.chat_message("assistant", avatar=":material/verified_user:"):
        t0 = time.time()
        msg = {"role": "assistant", "content": "", "table": None, "meta": None}
        try:
            svc, loc = parse_query(q)
            asked_place, llm_failed = None, False
            if svc is None or loc is None:          # anything our word lists missed -> ask the language model
                try:
                    with st.spinner("Understanding the question…"):
                        raw = llm([{"role": "system", "content": EXTRACT_PROMPT.format(services=", ".join(SERVICES))},
                                   {"role": "user", "content": q}], max_tokens=80, temperature=0)
                    m = re.search(r"\{.*\}", raw, re.S)
                    params = json.loads(m.group(0)) if m else {}
                except Exception:
                    params, llm_failed = {}, True   # model unavailable: carry on with what our word lists found
                svc = svc or (params.get("service") if params.get("service") in SERVICES else None)
                if loc is None and params.get("location"):
                    asked_place = str(params["location"])
                    loc = resolve_place(asked_place)
            used_default = loc is None
            if loc is None and llm_failed and any(ord(ch) > 127 for ch in q):
                msg["content"] = T("no_place_read")      # non-English place name and no model to read it: ask
                st.markdown(msg["content"])
            elif asked_place and loc is None:
                msg["content"] = T("no_place", place=esc(asked_place))
                st.markdown(msg["content"])
            elif svc is None:
                msg["content"] = T("no_service", services=", ".join(svc_label(x) for x in SERVICES))
                st.markdown(msg["content"])
            else:
                loc = loc or st.session_state.get("loc")
                res = rank(svc, float(loc["lat"]), float(loc["lon"]), 100)
                top = res[res["confidence"] != "LOW"].head(5)
                if not len(top):
                    top = res.head(5)
                meta = T("searchline", service=f"<b>{esc(svc_label(svc))}</b>",
                         place=f"<b>{esc(loc['desc'])}</b>" + (" " + T("default_loc") if used_default else ""),
                         n=len(top))
                st.markdown(f'<div class="searchline">{meta}</div>', unsafe_allow_html=True)
                answer_box = st.empty()
                if len(top):
                    msg["table"] = contact_table(top)
                    show_contacts(msg["table"])
                with answer_box.container():
                    try:
                        msg["content"] = st.write_stream(llm_stream(
                            [{"role": "system", "content": ANSWER_PROMPT.format(language=lang_name())},
                             {"role": "user", "content": f"QUESTION: {q}\nSEARCH: {svc} near {loc['desc']} (within "
                              f"100 km)\nFACILITY DATA:\n{facility_context(top) or 'No facilities found.'}"}]))
                    except Exception as e:
                        msg["content"] = f"*{T('llm_down')}* ({str(e)[:140]})"
                        st.markdown(msg["content"])
                msg["meta"] = meta + " · " + T("answered", s=f"{time.time() - t0:.1f}")
        except Exception as e:
            msg["content"] = f"Something went wrong ({str(e)[:160]}). You can still use “Find care”."
            st.markdown(msg["content"])
    st.session_state.chat.append(msg)


# ---------------------------------------------------------------- header + view switch
@st.cache_resource(ttl=3600, show_spinner=False)
def dataset_totals():
    """Headline numbers for the whole dataset (not only facilities that offer one of our 12 services)."""
    def count(sql_where, table, local_filter=None):
        if LOCAL_DIR:
            df = optional(table)
            return int(len(local_filter(df)) if local_filter is not None and len(df) else len(df)) or None
        try:
            return int(query(f"SELECT COUNT(*) AS n FROM {DB}.{table} {sql_where}")["n"].iloc[0])
        except Exception:
            return None
    records = count("", "facilities_clean")
    google = count("WHERE google_match IN ('strong', 'weak')", "facilities_verified",
                   lambda d: d[d["google_match"].isin(["strong", "weak"])])
    return records, google


n_records, n_google_all = dataset_totals()
n_fac = ev["unique_id"].nunique()                                   # facilities with evidence for a service
n_records = n_records or 10088                                      # the full Virtue Foundation dataset
n_strong = int((ev["confidence"] == "HIGH").sum())
n_google = n_google_all or int(ev.drop_duplicates("unique_id")["google_match"].isin(["strong", "weak"]).sum())


def hero(view):
    brand = f'<div class="brand">{LOGO.format(s=30, cls="")}VeriCare</div>'
    intl = "" if lang_name() == "English" else " intl"
    if view == "find":
        stats = "".join(f'<div class="stat"><b>{v}</b><span>{esc(T(k))}</span></div>' for v, k in [
            (f"{n_records:,}", "stat_fac"), (f"{n_fac:,}", "stat_ev"), (f"{n_strong:,}", "stat_strong"),
            (f"{n_google:,}", "stat_google")])
        body = (f'<div class="bottom"><h1 class="display">{T("headline")}</h1>'
                f'<div class="side"><p>{esc(T("tagline"))}</p><div class="stats">{stats}</div></div></div>')
        st.markdown(f'<div class="hero big{intl}">{DISCS}{brand}{body}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="hero small{intl}">{DISCS}{brand}<div class="bottom"><h1 class="display">'
                    f'{esc(T("nav_" + view))}</h1></div></div>', unsafe_allow_html=True)


nav, langbox = st.columns([5, 1.1], vertical_alignment="center")
with langbox:
    st.selectbox(T("language"), list(LANGS), key="lang", label_visibility="collapsed",
                 help="Language for the app and the chat answers / ऐप और चैट की भाषा")
with nav:                       # one widget per language, so switching language keeps you on the same tab
    VIEWS = ["find", "ask", "gaps", "ins"]
    labels = [T(f"nav_{k}") for k in VIEWS]
    cur = st.session_state.get("view_k", "find")
    choice = st.radio("View", labels, index=VIEWS.index(cur) if cur in VIEWS else 0, horizontal=True,
                      label_visibility="collapsed", key=f"view_{lang_name()}")
    view = VIEWS[labels.index(choice)] if choice in labels else "find"
    st.session_state.view_k = view
hero(view)
if view == "ask":
    render_ask()
    st.stop()
if view == "gaps":
    render_gaps()
    st.stop()
if view == "ins":
    render_insights()
    st.stop()

with st.container(border=True):
    c = st.columns([2.2, 3.2, 1.6, 1.2])
    cur_svc = st.session_state.get("service_k", "Dialysis")
    svc_labels = [svc_label(x) for x in SERVICES]           # one widget per language (labels change with it)
    pick = c[0].selectbox(T("care_needed"), svc_labels, key=f"service_{lang_name()}",
                          index=SERVICES.index(cur_svc) if cur_svc in SERVICES else 0)
    service = SERVICES[svc_labels.index(pick)] if pick in svc_labels else SERVICES[0]
    st.session_state.service_k = service
    sel = c[1].selectbox(T("patient_location"), LOC_LABELS, index=None, key="place_sel",
                         placeholder=T("loc_placeholder"))
    max_km = c[2].slider(T("radius"), 10, 300, 100, step=10)
    show_weak = c[3].toggle(T("weak_leads"), value=False, help=T("weak_help"))
    with st.expander(T("coords")):
        cc = st.columns([3, 1])
        txt = cc[0].text_input("latitude, longitude", placeholder="e.g. 25.5941, 85.1376", label_visibility="collapsed")
        if cc[1].button(T("use_these")) and txt:
            m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", txt)
            if m:
                st.session_state.loc = {"lat": float(m.group(1)), "lon": float(m.group(2)),
                                        "desc": f"Coordinates {m.group(1)}, {m.group(2)}", "note": None}
            else:
                st.error("Use the format: 25.5941, 85.1376")

if sel and sel != st.session_state.get("applied_sel"):
    st.session_state.applied_sel = sel
    st.session_state.loc = LOC_OPTS[sel]

loc = st.session_state.loc
ulat, ulon = float(loc["lat"]), float(loc["lon"])
res = rank(service, ulat, ulon, max_km)
n_high, n_med, n_low = [(res["confidence"] == k).sum() for k in ("HIGH", "MEDIUM", "LOW")]
shown = res if show_weak else res[res["confidence"] != "LOW"]
top = shown.head(8)

st.markdown(
    f'<div class="summary">{icon("pin")}<b>{esc(loc["desc"])}</b>'
    + (f' <span style="color:#7a6a64">({esc(loc["note"])})</span>' if loc.get("note") else "")
    + f' &nbsp;·&nbsp; {esc(T("within", km=max_km))}: '
      f'<span class="pill pill-HIGH">{esc(T("n_strong", n=n_high))}</span>'
      f'<span class="pill pill-MEDIUM">{esc(T("n_some", n=n_med))}</span>'
      f'<span class="pill pill-LOW">{esc(T("n_weak", n=n_low))}{"" if show_weak else " " + esc(T("hidden"))}</span></div>',
    unsafe_allow_html=True)

if n_high == 0:
    allf = rank(service, ulat, ulon, 3000)
    near = allf[allf["confidence"] == "HIGH"].sort_values("distance_km").head(1)
    msg = T("no_strong", km=max_km)
    if len(near):
        n = near.iloc[0]
        msg += " " + T("nearest_strong", name=f"**{n['name']}, {n['city']}**", km=f"{n['distance_km']:.0f}")
    st.error(msg)

left, right = st.columns([5, 4], gap="medium")
with right:
    fmap = folium.Map(location=[ulat, ulon], zoom_start=10, tiles="OpenStreetMap", control_scale=True)
    folium.Marker([ulat, ulon], tooltip="Patient location",
                  icon=folium.Icon(color="blue", icon="user", prefix="fa")).add_to(fmap)
    folium.Circle([ulat, ulon], radius=max_km * 1000, color=BRAND, weight=1, fill=True, fill_opacity=0.03).add_to(fmap)
    pts_ = [[ulat, ulon]]
    for i, (_, r) in enumerate(top.iterrows(), 1):
        col = CONF_COLOR[r["confidence"]]
        html = (f'<div style="background:{col};color:white;border:2px solid white;border-radius:10px;'
                f'width:28px;height:28px;line-height:24px;text-align:center;font-weight:800;font-family:Inter,sans-serif;'
                f'box-shadow:0 2px 8px rgba(0,0,0,.35)">{i}</div>')
        popup = (f"<b>{i}. {esc(r['name'])}</b><br>{r['distance_km']:.1f} km · {T('conf_' + r['confidence'])}"
                 f"<br>Tel: {esc(fmt_phone(r['phone']) or 'not listed')}")
        folium.Marker([r["loc_lat"], r["loc_lon"]], icon=folium.DivIcon(html=html, icon_size=(28, 28), icon_anchor=(14, 14)),
                      tooltip=f"{i}. {r['name']}", popup=folium.Popup(popup, max_width=260)).add_to(fmap)
        pts_.append([r["loc_lat"], r["loc_lon"]])
    if len(pts_) > 1:
        fmap.fit_bounds(pts_, padding=(30, 30))
    out = st_folium(fmap, height=620, use_container_width=True, returned_objects=["last_clicked"], key="map")
    if out and out.get("last_clicked"):
        lc = out["last_clicked"]
        click = (round(lc["lat"], 5), round(lc["lng"], 5))
        if click != st.session_state.get("applied_click"):
            st.session_state.applied_click = click
            st.session_state.loc = {"lat": click[0], "lon": click[1],
                                    "desc": f"Pinned spot ({click[0]:.4f}, {click[1]:.4f})", "note": "Dropped on the map"}
            st.rerun()
    st.caption(T("map_caption"))

with left:
    if not len(top):
        st.warning(T("no_results"))
    for i, (_, r) in enumerate(top.iterrows(), 1):
        card(i, r)

# ---------------------------------------------------------------- sidebar (collapsed by default)
with st.sidebar:
    st.header("About VeriCare")
    with st.expander("How the ranking works", expanded=True):
        st.markdown(
            "1. **Evidence:** each facility's procedures, equipment, capabilities, description and Google "
            "reviews are searched for the service. Procedures/equipment = 3 points; capabilities, description, "
            "reviews = 2; related specialty = 1. Negative mentions (“not available”, “referred”) don't count.\n"
            "2. **Confidence:** *strong* = stated in 2+ different fields. Lowered if the location is uncertain, "
            "the record looks like several facilities mixed together, or text may say it's not offered.\n"
            "3. **Ranking:** 50% service evidence + 33% size (beds) + 17% closeness, × confidence "
            "(1.0 / 0.85 / 0.6). The radius slider filters out anything too far away.\n"
            "4. **Size:** bed counts are *listed*, *stated in the description*, from the *hospital website*, or "
            "**predicted** by a model trained on facilities whose size is known. Predicted values are labelled "
            "and count ×0.8.\n"
            "5. **Location:** checked against postcode, city, address and Google Maps.")
    if len(size_metrics):
        with st.expander("How accurate are the predicted sizes?"):
            for _, m in size_metrics.iterrows():
                if m["target"] == "beds" and isnum(m.get("accuracy")):
                    st.markdown(
                        f"- **Beds** (under {int(m['bed_cut'])} vs {int(m['bed_cut'])}+, trained on "
                        f"{int(m['n_train']):,} facilities): **{m['shown_accuracy']:.0%} correct** on the "
                        f"{m['shown_share']:.0%} of facilities where a size is shown "
                        f"(balanced {m['shown_balanced_accuracy']:.0%}); {m['accuracy']:.0%} across all, "
                        f"vs {m['naive_accuracy']:.0%} for always guessing the bigger group.")
                elif isnum(m.get("median_abs_pct_error")):
                    st.markdown(
                        f"- **{m['target'].replace('_', ' ').title()}** (trained on {int(m['n_train']):,}): typical "
                        f"error **{m['median_abs_pct_error']:.0%}** (naive guess {m['baseline_abs_pct_error']:.0%}); "
                        f"real value inside the shown range {m['within_range_pct']:.0%} of the time.")
            if len(size_imp):
                ti = size_imp[size_imp["target"] == "beds"].head(6).copy()
                tot = ti["importance"].clip(lower=0).sum() or 1
                st.markdown("**What drives the bed-size prediction:** " + ", ".join(
                    f"{g} ({max(v, 0) / tot:.0%})" for g, v in zip(ti["group"], ti["importance"])))
            st.caption("Tested with 5-fold cross-validation: each facility is judged by a model that never saw it.")
    with st.expander("About the data"):
        st.markdown("10,088 facility records from the Virtue Foundation dataset, extracted from web pages. "
                    "Every field is treated as a **claim to check**, not a fact.")
        keep = report[report["check"].str.contains(
            "Location confidence|Merge risk: high|shared with 3|Doctor count of 1|Records usable|Missing", regex=True)]
        for _, r in keep.iterrows():
            n = pd.to_numeric(r["count"], errors="coerce")
            st.markdown(f"- **{int(n) if isnum(n) else 0:,}** {r['check']}")
