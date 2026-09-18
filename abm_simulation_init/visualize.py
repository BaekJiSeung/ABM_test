# %% ============================================================
# visualize_cpe_lambda_daily_3chart_save.py
#
# Mesa ModularServer visualization for:
# - model/cpe_model_month_lambda.py
# - model/agents_lambda.py
#
# Model file is NOT modified.
#
# Visualization:
# - Grid updates every tick.
# - Daily chart records one point per completed simulated day.
# - Chart shows only:
#     1. Current patients
#     2. Cumulative HCW-related infections
#     3. Cumulative imported CRE-positive inputs
#
# Save:
# - Press "Save ABM figure" in the browser.
# - It saves grid + daily graph + legend as:
#     abm_grid_structure.png
#
# Run:
# python visualize_cpe_lambda_daily_3chart_save.py
#
# Browser:
# http://127.0.0.1:8521
# ============================================================

import numpy as np

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, TextElement
from mesa.visualization.UserParam import Slider, Choice

from model.cpe_model_month_lambda import CPE_Model_month
from model.agents_lambda import Nurse, Dr, XrayDr


# ============================================================
# 1. Agent portrayal
# ============================================================

def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "Layer": 1,
        "r": 0.35,
    }

    # --------------------------------------------------------
    # Patient
    # --------------------------------------------------------
    if getattr(agent, "isPatient", False):

        portrayal["Shape"] = "circle"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 6
        portrayal["r"] = 0.32

        # CRE-positive / detected patient
        if getattr(agent, "positive", False):
            portrayal["Color"] = "#8e44ad"
            portrayal["text"] = "P"
            portrayal["text_color"] = "white"

        # Colonized patient
        elif getattr(agent, "colonized", False):
            portrayal["Color"] = "#de1616"

            # Imported colonized patient
            if getattr(agent, "preinfection", False):
                portrayal["text"] = "I"
                portrayal["text_color"] = "white"

        # Susceptible patient
        else:
            portrayal["Color"] = "#666666"

        # Isolated patient
        if getattr(agent, "isolated", False):
            portrayal["r"] = 0.38

        # Patient waiting to move to isolation
        if getattr(agent, "move2isol", False):
            portrayal["text"] = "M"
            portrayal["text_color"] = "yellow"

        return portrayal

    # --------------------------------------------------------
    # Nurse
    # --------------------------------------------------------
    if isinstance(agent, Nurse):

        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 8
        portrayal["w"] = 0.25
        portrayal["h"] = 0.25

        if getattr(agent, "colonized", False):
            portrayal["Color"] = "#ff0000"
            portrayal["text"] = "N"
            portrayal["text_color"] = "white"
        else:
            if getattr(agent, "hall", None) == 6:
                portrayal["Color"] = "#0ababa"
            else:
                portrayal["Color"] = "#096363"

        return portrayal

    # --------------------------------------------------------
    # Doctor / X-ray doctor
    # --------------------------------------------------------
    if isinstance(agent, Dr):

        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 8
        portrayal["w"] = 0.25
        portrayal["h"] = 0.25

        if getattr(agent, "colonized", False):
            portrayal["Color"] = "#ff0000"
            portrayal["text"] = "D"
            portrayal["text_color"] = "white"
        else:
            if isinstance(agent, XrayDr):
                portrayal["Color"] = "#000080"
                portrayal["text"] = "X"
                portrayal["text_color"] = "white"
            else:
                portrayal["Color"] = "#000000"

        return portrayal

    # --------------------------------------------------------
    # Bed / isolated bed
    # --------------------------------------------------------
    if getattr(agent, "isBed", False):

        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 1
        portrayal["w"] = 0.55
        portrayal["h"] = 0.90

        if getattr(agent, "isIsolatedBed", False):
            portrayal["Color"] = "#fdc4ac"
        else:
            portrayal["Color"] = "#fff2c7"

        if getattr(agent, "filledSick", False):
            portrayal["text"] = "+"
            portrayal["text_color"] = "red"
        elif getattr(agent, "filled", False):
            portrayal["text"] = "."
            portrayal["text_color"] = "black"

        return portrayal

    # --------------------------------------------------------
    # Goo / environment
    # --------------------------------------------------------
    if getattr(agent, "isGoo", False):

        portrayal["Shape"] = "rect"
        portrayal["Filled"] = "true"
        portrayal["Layer"] = 2
        portrayal["w"] = 0.75
        portrayal["h"] = 0.75

        if getattr(agent, "colonized", False):
            portrayal["Color"] = "#cfb574"
            portrayal["text"] = "G"
            portrayal["text_color"] = "black"
        else:
            portrayal["Color"] = "#9ccedb"

        return portrayal

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------
    portrayal["Shape"] = "circle"
    portrayal["Color"] = "#cccccc"
    portrayal["Layer"] = 0
    portrayal["r"] = 0.2

    return portrayal


# ============================================================
# 2. CanvasGrid
# ============================================================

grid = CanvasGrid(
    agent_portrayal,
    32,
    11,
    950,
    330,
)


# ============================================================
# 3. Daily three-line chart
# ============================================================

class DailyThreeLineChart(TextElement):
    """
    Visualization-only daily chart.

    This class does not use model.datacollector.
    Therefore, model/cpe_model_month_lambda.py does not need to be modified.

    It records one point per completed simulated day.

    Series:
    - Current patients
    - Cumulative HCW-related infections
    - Cumulative imported CRE-positive inputs
    """

    def __init__(self, max_days=600):
        super().__init__()

        self.max_days = max_days
        self.last_recorded_day = None
        self.history = []
        self.cached_html = ""

    def _make_polyline(self, values, x0, y0, width, height, y_max):

        if len(values) == 0:
            return ""

        if len(values) == 1:
            x = x0
            y = y0 + height - (values[0] / y_max) * height
            return f"{x:.1f},{y:.1f}"

        points = []

        for i, val in enumerate(values):
            x = x0 + width * i / (len(values) - 1)
            y = y0 + height - (val / y_max) * height
            points.append(f"{x:.1f},{y:.1f}")

        return " ".join(points)

    def _render_chart(self):

        if len(self.history) == 0:
            return """
            <div id="daily-chart-export" style="font-family:Arial; padding:8px;">
                <b>Daily ABM summary</b><br>
                Waiting for the first completed simulated day...
            </div>
            """

        data = self.history[-self.max_days:]

        days = [
            row["day"]
            for row in data
        ]

        current_patients = [
            row["current_patients"]
            for row in data
        ]

        cumulative_hcw_related_infections = [
            row["cumulative_hcw_related_infections"]
            for row in data
        ]

        cumulative_imported_inputs = [
            row["cumulative_imported_inputs"]
            for row in data
        ]

        all_values = (
            current_patients
            + cumulative_hcw_related_infections
            + cumulative_imported_inputs
        )

        y_max = max(max(all_values), 1)
        y_max = int(np.ceil(y_max + 1))

        x0 = 55
        y0 = 20
        width = 780
        height = 210

        p_patients = self._make_polyline(
            current_patients,
            x0,
            y0,
            width,
            height,
            y_max,
        )

        p_hcw_inf = self._make_polyline(
            cumulative_hcw_related_infections,
            x0,
            y0,
            width,
            height,
            y_max,
        )

        p_imported = self._make_polyline(
            cumulative_imported_inputs,
            x0,
            y0,
            width,
            height,
            y_max,
        )

        day_start = days[0]
        day_end = days[-1]

        y_mid = y0 + height / 2
        y_bottom = y0 + height

        latest_current_patients = current_patients[-1]
        latest_hcw_inf = cumulative_hcw_related_infections[-1]
        latest_imported = cumulative_imported_inputs[-1]

        html = f"""
        <div id="daily-chart-export" style="font-family:Arial; padding:8px; width:930px;">
            <div style="font-size:16px; margin-bottom:4px;">
                <b>Daily ABM summary</b>
                <span style="font-size:12px; color:#555;">
                    one point per simulated day, days {day_start}--{day_end}
                </span>
            </div>

            <svg width="900" height="295" style="border:1px solid #ddd; background:#fafafa;">

                <!-- Axes -->
                <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" stroke="#333" stroke-width="1"/>
                <line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" stroke="#333" stroke-width="1"/>

                <!-- Grid lines -->
                <line x1="{x0}" y1="{y0}" x2="{x0 + width}" y2="{y0}" stroke="#e0e0e0"/>
                <line x1="{x0}" y1="{y_mid}" x2="{x0 + width}" y2="{y_mid}" stroke="#e0e0e0"/>
                <line x1="{x0}" y1="{y_bottom}" x2="{x0 + width}" y2="{y_bottom}" stroke="#e0e0e0"/>

                <!-- Y labels -->
                <text x="10" y="{y0 + 5}" font-size="11">{y_max}</text>
                <text x="10" y="{y_mid + 5}" font-size="11">{y_max / 2:.1f}</text>
                <text x="10" y="{y_bottom + 5}" font-size="11">0</text>

                <!-- X labels -->
                <text x="{x0}" y="{y0 + height + 20}" font-size="11">Day {day_start}</text>
                <text x="{x0 + width - 55}" y="{y0 + height + 20}" font-size="11">Day {day_end}</text>

                <!-- Lines -->
                <polyline points="{p_patients}" fill="none" stroke="#666666" stroke-width="2.5"/>
                <polyline points="{p_hcw_inf}" fill="none" stroke="#000000" stroke-width="2.5"/>
                <polyline points="{p_imported}" fill="none" stroke="#8e44ad" stroke-width="2.5"/>

                <!-- Legend -->
                <rect x="60" y="245" width="10" height="10" fill="#666666"/>
                <text x="75" y="254" font-size="12">Current patients: {latest_current_patients}</text>

                <rect x="250" y="245" width="10" height="10" fill="#000000"/>
                <text x="265" y="254" font-size="12">Cumulative HCW-related infections: {latest_hcw_inf}</text>

                <rect x="560" y="245" width="10" height="10" fill="#8e44ad"/>
                <text x="575" y="254" font-size="12">Cumulative imported inputs: {latest_imported}</text>
            </svg>
        </div>
        """

        return html

    def render(self, model):

        # ----------------------------------------------------
        # Use model.history if available.
        # The model records daily states at the end of each day.
        # This avoids modifying the model file.
        # ----------------------------------------------------
        hist = getattr(model, "history", None)

        if hist is not None and "day" in hist and len(hist["day"]) > 0:

            current_day = int(hist["day"][-1])

            # Record only once per completed simulated day
            if self.last_recorded_day != current_day:

                row = {
                    "day": current_day,

                    # 1. Current patients
                    "current_patients": len(
                        getattr(model, "current_patients", [])
                    ),

                    # 2. Cumulative HCW-related infections
                    "cumulative_hcw_related_infections": getattr(
                        model,
                        "cumul_sick_patients_by_HCW",
                        0,
                    ),

                    # 3. Cumulative imported CRE-positive inputs
                    # model.P_I is the cumulative imported patient counter.
                    "cumulative_imported_inputs": getattr(
                        model,
                        "P_I",
                        0,
                    ),
                }

                self.history.append(row)
                self.last_recorded_day = current_day
                self.cached_html = self._render_chart()

            return self.cached_html

        # ----------------------------------------------------
        # Before the first full day is completed
        # ----------------------------------------------------
        return """
        <div id="daily-chart-export" style="font-family:Arial; padding:8px;">
            <b>Daily ABM summary</b><br>
            Waiting for the first completed simulated day...
        </div>
        """


# ============================================================
# 4. Text elements
# ============================================================

class TickCounter(TextElement):
    def render(self, model):

        ticks_in_hour = getattr(model, "ticks_in_hour", 108)
        ticks_in_day = getattr(model, "ticks_in_day", 108 * 24)

        # model.schedule.time is reset by the model every day.
        current_tick = model.schedule.time

        current_day = getattr(
            model,
            "day",
            int(current_tick // ticks_in_day),
        )

        tick_in_day = current_tick % ticks_in_day

        current_hour = tick_in_day // ticks_in_hour
        tick_in_hour = tick_in_day % ticks_in_hour

        return (
            f"<b>Time</b> | "
            f"DAY: {int(current_day)} | "
            f"HOUR: {int(current_hour)} | "
            f"TICK: {int(tick_in_hour)}/{ticks_in_hour} | "
            f"Current patients: {len(getattr(model, 'current_patients', []))} | "
            f"Cumulative patients: {getattr(model, 'cumul_patients', 0)} | "
            f"Cumulative colonized patients: {getattr(model, 'cumul_sick_patients', 0)} | "
            f"Cumulative HCW-related infections: {getattr(model, 'cumul_sick_patients_by_HCW', 0)} | "
            f"Cumulative imported inputs: {getattr(model, 'P_I', 0)}"
        )


class LegendElement(TextElement):
    def render(self, model):
        return """
        <div id="legend-export" style="font-size: 14px; line-height: 1.8; padding: 8px;">
            <b>Legend</b><br>

            <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:#666666; margin-right:6px;"></span>
            Susceptible patient<br>

            <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:#de1616; margin-right:6px;"></span>
            Colonized patient<br>

            <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:#8e44ad; margin-right:6px;"></span>
            CRE-positive patient / imported input<br>

            <span style="display:inline-block; width:14px; height:14px; background:#0ababa; margin-right:6px;"></span>
            Nurse / HCW<br>

            <span style="display:inline-block; width:14px; height:14px; background:#000000; margin-right:6px;"></span>
            Doctor<br>

            <span style="display:inline-block; width:14px; height:14px; background:#000080; margin-right:6px;"></span>
            X-ray doctor<br>

            <span style="display:inline-block; width:14px; height:14px; background:#ff0000; margin-right:6px;"></span>
            Colonized HCW<br>

            <span style="display:inline-block; width:14px; height:14px; background:#fff2c7; border:1px solid #999; margin-right:6px;"></span>
            Shared bed<br>

            <span style="display:inline-block; width:14px; height:14px; background:#fdc4ac; border:1px solid #999; margin-right:6px;"></span>
            Isolated bed<br>

            <span style="display:inline-block; width:14px; height:14px; background:#9ccedb; border:1px solid #999; margin-right:6px;"></span>
            Clean environment<br>

            <span style="display:inline-block; width:14px; height:14px; background:#cfb574; border:1px solid #999; margin-right:6px;"></span>
            Contaminated environment
        </div>
        """


class SaveFigureElement(TextElement):
    def render(self, model):
        return r"""
        <div style="padding:8px;">
            <button
                type="button"
                style="
                    padding:8px 14px;
                    font-size:14px;
                    cursor:pointer;
                    border:1px solid #999;
                    background:#f5f5f5;
                    border-radius:4px;
                "
                onclick="
                    const gridCanvas = document.querySelector('canvas');
                    const chartSvg = document.querySelector('#daily-chart-export svg');

                    if (!gridCanvas) {
                        alert('Grid canvas not found.');
                    } else if (!chartSvg) {
                        alert('Daily chart is not ready yet. Run at least one simulated day.');
                    } else {
                        const padding = 25;
                        const titleH = 40;
                        const gap = 18;

                        const gridW = gridCanvas.width;
                        const gridH = gridCanvas.height;

                        const chartW = 900;
                        const chartH = 295;

                        const legendH = 120;

                        const outW = Math.max(gridW, chartW) + padding * 2;
                        const outH = padding + titleH + gridH + gap + chartH + gap + legendH + padding;

                        const outCanvas = document.createElement('canvas');
                        outCanvas.width = outW;
                        outCanvas.height = outH;

                        const ctx = outCanvas.getContext('2d');

                        ctx.fillStyle = 'white';
                        ctx.fillRect(0, 0, outW, outH);

                        ctx.fillStyle = 'black';
                        ctx.font = 'bold 20px Arial';
                        ctx.fillText('ABM visualization and daily output summary', padding, 30);

                        const gridX = padding;
                        const gridY = padding + titleH;
                        ctx.drawImage(gridCanvas, gridX, gridY);

                        const svgText = new XMLSerializer().serializeToString(chartSvg);
                        const svgBlob = new Blob([svgText], {type: 'image/svg+xml;charset=utf-8'});
                        const url = URL.createObjectURL(svgBlob);

                        const chartImg = new Image();

                        chartImg.onload = function() {
                            const chartX = padding;
                            const chartY = gridY + gridH + gap;
                            ctx.drawImage(chartImg, chartX, chartY);

                            URL.revokeObjectURL(url);

                            const legendY = chartY + chartH + gap + 10;

                            ctx.font = 'bold 14px Arial';
                            ctx.fillStyle = 'black';
                            ctx.fillText('Legend', padding, legendY);

                            function legendItem(x, y, color, text, circle=false) {
                                ctx.fillStyle = color;

                                if (circle) {
                                    ctx.beginPath();
                                    ctx.arc(x + 7, y - 5, 7, 0, 2 * Math.PI);
                                    ctx.fill();
                                } else {
                                    ctx.fillRect(x, y - 13, 14, 14);
                                }

                                ctx.fillStyle = 'black';
                                ctx.font = '13px Arial';
                                ctx.fillText(text, x + 22, y);
                            }

                            legendItem(padding, legendY + 28, '#666666', 'Susceptible patient', true);
                            legendItem(padding + 230, legendY + 28, '#de1616', 'Colonized patient', true);
                            legendItem(padding + 460, legendY + 28, '#8e44ad', 'CRE-positive / imported', true);

                            legendItem(padding, legendY + 55, '#0ababa', 'Nurse / HCW');
                            legendItem(padding + 230, legendY + 55, '#000000', 'Doctor');
                            legendItem(padding + 460, legendY + 55, '#000080', 'X-ray doctor');

                            legendItem(padding, legendY + 82, '#ff0000', 'Colonized HCW');
                            legendItem(padding + 230, legendY + 82, '#fff2c7', 'Shared bed');
                            legendItem(padding + 460, legendY + 82, '#fdc4ac', 'Isolated bed');

                            legendItem(padding, legendY + 109, '#9ccedb', 'Clean environment');
                            legendItem(padding + 230, legendY + 109, '#cfb574', 'Contaminated environment');

                            const link = document.createElement('a');
                            link.download = 'abm_grid_structure.png';
                            link.href = outCanvas.toDataURL('image/png');
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                        };

                        chartImg.onerror = function() {
                            URL.revokeObjectURL(url);
                            alert('Failed to render chart image.');
                        };

                        chartImg.src = url;
                    }
                "
            >
                Save ABM figure
            </button>
        </div>
        """


daily_chart = DailyThreeLineChart(max_days=600)
tick_counter = TickCounter()
legend_element = LegendElement()
save_button = SaveFigureElement()


# ============================================================
# 5. Model parameters
# ============================================================

model_params = {
    "data_type": Choice(
        "Data type",
        value="A",
        choices=["A", "B", "B_"],
    ),

    "prob_new_patient": Slider(
        "Probability of admission of new patient",
        value=0.003,
        min_value=0.000,
        max_value=0.100,
        step=0.001,
        description="Probability that an empty bed receives a new patient.",
    ),

    "prob_transmission": Slider(
        "Probability of transmission",
        value=0.03847,
        min_value=0.000,
        max_value=0.200,
        step=0.0001,
        description="Transmission probability. This plays the role of beta_ABM.",
    ),

    "isolation_factor": Slider(
        "Isolation factor",
        value=0.75,
        min_value=0.00,
        max_value=1.00,
        step=0.05,
        description="Transmission reduction factor for isolated patients.",
    ),

    "cleaningDay": Slider(
        "Days before environmental cleaning",
        value=180,
        min_value=1,
        max_value=360,
        step=1,
        description="Cleaning interval for environmental contamination.",
    ),

    "hcw_wash_rate": Slider(
        "Handwash probability",
        value=0.90,
        min_value=0.00,
        max_value=1.00,
        step=0.05,
        description="Probability that a healthcare worker decolonizes after contact.",
    ),

    "isolation_time": Slider(
        "Isolation delay for colonized patients",
        value=14,
        min_value=1,
        max_value=30,
        step=1,
        description="Delay until a colonized patient is moved to isolation.",
    ),

    "init_env": Slider(
        "Initial contaminated environment count",
        value=9,
        min_value=0,
        max_value=30,
        step=1,
        description="Number of initially contaminated environmental reservoirs.",
    ),

    "tau_offset_days": Slider(
        "Cleaning offset days",
        value=140,
        min_value=0,
        max_value=360,
        step=1,
        description="Offset for cleaning schedule.",
    ),

    "height": 11,
    "width": 32,
}


# ============================================================
# 6. Server modules
# ============================================================

server_modules = [
    grid,
    daily_chart,
    tick_counter,
    legend_element,
    save_button,
]


# ============================================================
# 7. Server
# ============================================================

server = ModularServer(
    CPE_Model_month,
    server_modules,
    "CPE/CRE ABM Daily Visualization",
    model_params,
)

server.port = 8523


# ============================================================
# 8. Launch
# ============================================================

server.launch()