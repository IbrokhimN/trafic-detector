from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box as richbox

from traffic_light import calc_green_time

console = Console()

def build_console_dashboard(
    vehicle_count, ped_count, has_elderly, emergency,
    avg_speed, count_up, count_down, frame_skipped,
    incidents, phase_name, fps, frame_num, elapsed,
    eco_co2, eco_fuel, roi_active
):
    timing = calc_green_time(vehicle_count, ped_count, avg_speed, has_elderly, emergency)
    v_green = timing["v_green"]
    p_green = timing["p_green"]

    main = Table(
        title="SMART TRAFFIC POWERED BY AI",
        box=richbox.DOUBLE_EDGE,
        title_style="bold white",
        border_style="cyan",
        show_header=True,
        header_style="bold white",
        expand=True,
        padding=(0, 1),
    )
    main.add_column("Metric", style="dim", width=28)
    main.add_column("Value", justify="right", width=20)

    rows = [
        ("Vehicles in zone", f"[bold green]{vehicle_count}[/]"),
        ("Pedestrians", f"[bold cyan]{ped_count}[/]"),
        ("Elderly detected", "[bold yellow]YES[/]" if has_elderly else "[dim]no[/]"),
        ("Emergency", "[bold red blink]YES[/]" if emergency else "[dim]—[/]"),
        ("Avg speed", f"[bold yellow]{avg_speed:.1f}[/] km/h"),
        ("Crossed up", f"[green]{count_up}[/]"),
        ("Crossed down", f"[red]{count_down}[/]"),
        ("Total crossed", f"[bold white]{count_up + count_down}[/]"),
        ("Skipped (outside)", f"[dim]{frame_skipped}[/]"),
        ("Incidents", f"[bold red]{incidents}[/]" if incidents > 0 else "[dim]0[/]"),
        ("Phase", f"[cyan]{phase_name}[/]"),
        ("Frame", f"[dim]{frame_num}[/]"),
        ("Elapsed", f"[dim]{elapsed:.0f}s[/]"),
        ("FPS", f"[dim]{fps:.0f}[/]"),
        ("CO2 saved", f"[green]{eco_co2:.0f}[/] g"),
        ("Fuel saved", f"[green]{eco_fuel:.0f}[/] ml"),
    ]

    for label, value in rows:
        main.add_row(label, value)

    timing_table = Table(
        title="traffic light recommendation",
        box=richbox.ROUNDED,
        border_style="green",
        expand=True,
        show_header=False,
        padding=(0, 1),
    )
    timing_table.add_column("direction", width=20)
    timing_table.add_column("time", justify="center", width=10)
    timing_table.add_column("reason", width=40)

    timing_table.add_row("vehicles", f"{v_green}s", f"queue ~{timing['v_clearance']}s" if v_green > 0 else "no vehicles")
    timing_table.add_row("pedestrians", f"{p_green}s", f"crossing {timing['p_crossing']}s" if p_green > 0 else "no pedestrians")
    timing_table.add_row("full cycle", f"{timing['cycle']}s", timing['efficiency'])

    roi_text = "[bold cyan]ROI active[/] — counting only objects inside road zone" if roi_active else "[dim]ROI not set — whole frame[/]"

    output = Table.grid(expand=True)
    output.add_row(main)
    output.add_row("")
    output.add_row(timing_table)
    output.add_row("")
    output.add_row(Panel(f"[bold]{timing['reason']}[/]", title="AI Decision", border_style="yellow", expand=True))
    output.add_row(f"  {roi_text}")

    return output

def print_session_summary(elapsed, frame_num, vehicle_db, total_pedestrians_seen,
                          count_up, count_down, skipped_outside, eco, max_density):
    console.print()
    console.print(Panel(
        "[bold white]session complete[/]",
        border_style="cyan",
        expand=True,
    ))

    t = Table(box=richbox.DOUBLE_EDGE, border_style="cyan", expand=True, show_header=False)
    t.add_column("Metric", style="bold", width=30)
    t.add_column("Value", justify="right", width=20)

    rows = [
        ("Duration",               f"{elapsed:.0f} sec"),
        ("Frames processed",       str(frame_num)),
        ("Unique vehicles",        str(len(vehicle_db.db))),
        ("Unique pedestrians",     str(len(total_pedestrians_seen))),
        ("Crossed UP",             str(count_up)),
        ("Crossed DOWN",           str(count_down)),
        ("Total crossed",          str(count_up + count_down)),
        ("Max density",            str(max_density)),
        ("Skipped (outside ROI)",  str(skipped_outside)),
        ("CO2 saved (est)",        f"{eco.co2_saved:.0f} g"),
        ("Fuel saved (est)",       f"{eco.fuel_saved * 1000:.0f} ml"),
    ]

    for metric, value in rows:
        t.add_row(metric, f"[bold cyan]{value}[/]")

    console.print(t)

    if vehicle_db.stats:
        vt = Table(title="Vehicle Types", box=richbox.ROUNDED, border_style="green", expand=True)
        vt.add_column("Type", style="bold")
        vt.add_column("Count", justify="right")
        for cls, cnt in sorted(vehicle_db.stats.items(), key=lambda x: -x[1]):
            vt.add_row(cls, f"[green]{cnt}[/]")
        console.print(vt)

    console.print(Panel(
        f"[bold]CSV:[/]  {CSV_LOG}\n[bold]JSON:[/] {JSON_DASHBOARD}",
        title="Output Files",
        border_style="blue",
    ))
    console.print()
