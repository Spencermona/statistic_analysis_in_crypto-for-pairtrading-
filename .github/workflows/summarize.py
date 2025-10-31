import json, datetime, os, sys

def main():
    msg_lines = []
    multi = 'signals_multi.json'
    single = 'signal_latest.json'
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    msg_lines.append(f"Signals summary @ {now} (Asia/Shanghai)")
    if os.path.exists(multi):
        try:
            with open(multi,'r',encoding='utf-8') as f:
                data = json.load(f)
            for base, info in data.items():
                pair = info.get('pair') or f"{info.get('cand','?')}-{info.get('base','?')}"
                z = info.get('z_now')
                notional = info.get('usd_notional') or info.get('usd_size') or 0.0
                mode = info.get('sizing_mode') or 'n/a'
                hl = info.get('half_life_days')
                try:
                    z_str = f"{float(z):.2f}" if z is not None else "n/a"
                except Exception:
                    z_str = str(z)
                try:
                    n_str = f"${float(notional):.2f}"
                except Exception:
                    n_str = str(notional)
                msg_lines.append(f"[{base}] {pair} | z={z_str} | notional={n_str} | hl={hl}d | mode={mode}")
        except Exception as e:
            msg_lines.append(f"(failed to parse {multi}: {e})")
    elif os.path.exists(single):
        try:
            with open(single,'r',encoding='utf-8') as f:
                info = json.load(f)
            pair = info.get('pair') or f"{info.get('cand','?')}-{info.get('base','?')}"
            z = info.get('z_now')
            notional = info.get('usd_notional') or info.get('usd_size') or 0.0
            mode = info.get('sizing_mode') or 'n/a'
            hl = info.get('half_life_days')
            try:
                z_str = f"{float(z):.2f}" if z is not None else "n/a"
            except Exception:
                z_str = str(z)
            try:
                n_str = f"${float(notional):.2f}"
            except Exception:
                n_str = str(notional)
            msg_lines.append(f"{pair} | z={z_str} | notional={n_str} | hl={hl}d | mode={mode}")
        except Exception as e:
            msg_lines.append(f"(failed to parse {single}: {e})")
    else:
        msg_lines.append("No signals JSON found.")

    text = "\n".join(msg_lines)
    print(text)
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/summary.txt','w',encoding='utf-8') as f:
        f.write(text)

if __name__ == '__main__':
    sys.exit(main())
