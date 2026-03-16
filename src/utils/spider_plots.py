import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.patheffects as pe


from plot_config import BLUE, RED


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if len(x) > 0 and x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, **kwargs):
            self.set_thetagrids(np.degrees(theta), labels, **kwargs)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    """Some Example data"""
    data = [
        ('CLIP', [
            [0.88, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2],
            [0.7, 0.2, 0.3, 0.2, 0.1, 0.12, 0.1, 0.2],
        ],
         {
             "start": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4],
             "end": [0.9, 0.8, 0.6, 0.6, 0.4, 0.4, 0.6, 0.9],
         },
         ),
        ('BioMedCLIP', [
            [0.2, 0.2, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        ],
         {
             "start": [0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
             "end": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
         },
        ),
    ]
    return data


def plot_intervals(
    ax,
    theta: np.ndarray,
    start: list[float] | np.ndarray,
    end: list[float] | np.ndarray,
    *,
    color=BLUE.hex,
    alpha=0.5,
    edgecolor=BLUE.hex,
    linewidth=0.5,
    zorder=2,
    connect=False,
    band_alpha=0.2,
    thickness: float = 0.05, 
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Plot intervals on each radar spoke either as polar bars (default) or as
    constant-thickness rectangles (equal_thickness=True).
    """

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    theta = np.asarray(theta, dtype=float)
    assert len(theta) == len(start) == len(end), "theta, start, end must have same length"

    # ylim must be set before we query ax.get_rmax() for dynamic thickness scaling
    if ylim is not None:
         ax.set_ylim(*ylim)
         
    if ylim == (0.0, 1.0):
         # 1. Draw grid lines but hide the default, misaligned labels
         ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], labels=[])
         
         # 2. Draw custom labels perfectly aligned with the polygon edge
         N = len(theta)
         # Pick an angle exactly halfway between the first two axes (e.g., Top-Left)
         label_angle_rad = (theta[0] + theta[1]) / 2.0
         
         # Polygons are closer to the center between axes by a factor of cos(pi/N)
         scale_factor = np.cos(np.pi / N)
         
         for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
             ax.text(
                 label_angle_rad, 
                 r * scale_factor,  # Scale the radius down to hit the flat line
                 str(r), 
                 ha='center', 
                 va='center', 
                 fontsize=8.5, 
                 color="#555555",
                 path_effects=[pe.withStroke(linewidth=3, foreground='white')],
                 zorder=100
             )

    # draw bars as polar bars (default)
    thickness = thickness * ax.get_rmax()
    for t, r0, r1 in zip(theta, start, end):
        # ensure order inner→outer for consistent orientation
        _fill_constant_width_rect(
            ax, t, r0, r1, thickness,
            facecolor=color, alpha=alpha,
            edgecolor=edgecolor if edgecolor is not None else "none",
            linewidth=linewidth, zorder=zorder
        )

    if connect:
        theta = np.asarray(theta, float)
        start = np.asarray(start, float)
        end = np.asarray(end, float)

        # clip to current r-limits to avoid odd triangles
        rmin, rmax = ax.get_ylim()
        start_c = np.clip(start, rmin, rmax)
        end_c = np.clip(end, rmin, rmax)

        # --- Wrap angles past 2π so the "end" boundary is continuous at the seam ---
        two_pi = 2.0 * np.pi
        theta_ext = np.r_[theta, theta[0] + two_pi]  # strictly increasing
        end_ext = np.r_[end_c, end_c[0]]
        start_ext = np.r_[start_c, start_c[0]]

        # Build polygon: forward along end, then backward along start
        tt = np.r_[theta_ext, theta_ext[::-1]]
        rr = np.r_[end_ext, start_ext[::-1]]

        ax.fill(tt, rr, facecolor=color, alpha=band_alpha, edgecolor="none", zorder=zorder - 1)

def _fill_constant_width_rect(ax, t, r0, r1, thickness, **kwargs):
    # points on the spoke in Cartesian
    c, s = np.cos(t), np.sin(t)
    p0 = np.array([r0 * c, r0 * s])
    p1 = np.array([r1 * c, r1 * s])

    # unit tangent (perpendicular to the radius)
    u_tan = np.array([-s, c])
    offset = (thickness / 2.0) * u_tan

    # rectangle corners in Cartesian
    q0a = p0 - offset
    q0b = p0 + offset
    q1a = p1 + offset
    q1b = p1 - offset

    # convert back to polar (theta, r) for a PolarAxes
    def cart2polar(xy):
        x, y = xy
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        return th, r

    corners = [q0a, q0b, q1a, q1b]
    ths, rs = zip(*[cart2polar(q) for q in corners])

    ax.fill(ths, rs, **kwargs)


def spider_plot(
    data: dict[str, dict[str, list[float] | dict[str, list[float]]]],
    class_labels: list[str],
    colors: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    *,
    show: bool = True,
    interval_thickness: float = 0.05,
    fig_size=(9, 4),
    show_profiles: bool = False,
    ylim: tuple[float, float] | None = None,
    chinese_labels: bool = False
) -> tuple[plt.Figure, plt.Axes] | None:
    """Create a spider (radar) plot with one subplot per data entry."""

    if markers is None:
        markers = {}
    if colors is None:
        colors = {
            label: color for label, color in zip(
                data[next(iter(data))].keys(),
                plt.rcParams['axes.prop_cycle'].by_key()['color']
            )
        }

    n_plots = len(data.keys())
    N = len(class_labels)
    theta = radar_factory(N, frame='polygon')
    fig, axs = plt.subplots(figsize=fig_size, nrows=1, ncols=n_plots, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.4)

    # plot each spider plot
    for ax_id, (data_name, data_plot) in enumerate(data.items()):
        ax = axs[ax_id] if n_plots > 1 else axs
        ax.set_title(data_name, weight='bold', size='large')

        ax.set_rlabel_position(45)

        for label, data_to_plot in data_plot.items():
            color = colors.get(label, "black")
            marker = markers.get(label, None)
            if isinstance(data_to_plot, dict):
                # plot upper/lower intervals
                alpha = 0.5
                plot_intervals(
                    ax,
                    theta,
                    start=data_to_plot["start"],
                    end=data_to_plot["end"],
                    connect=False,
                    alpha=alpha,
                    thickness=interval_thickness,
                    color=color,
                    ylim=ylim,
                )
                ax.fill([], [], label=label, facecolor=color, alpha=alpha)
            else:
                # plot line / dots
                ax.plot(
                    theta,
                    data_to_plot,
                    color=color,
                    marker=marker,
                    linewidth=0,
                    zorder=91,
                    markeredgecolor="white",  # marker outline
                    markeredgewidth=1.5,
                    markersize=10,
                    clip_on=False
                )
                if show_profiles:
                    line, = ax.plot(
                        theta,
                        data_to_plot,
                        color=color,
                        linewidth=1,
                        zorder=90,
                    )
                    # Add a white outline around the line
                    line.set_path_effects([
                        pe.Stroke(linewidth=3, foreground="white"),  # outline
                        pe.Normal()  # original line on top
                    ])
                    ax.plot([], [], label=label, c=color, marker=marker, linewidth=1, markersize=10, markeredgecolor="white", markeredgewidth=1.5)
                else:
                    ax.scatter([], [], label=label, c=color, marker=marker, edgecolor="white", linewidth=1.5, s=100)

        # add labels to each spoke the first class (top) usually fits better without breaks
        class_labels = [class_labels[0]] + [lbl.replace(" ", "\n") for lbl in class_labels[1:]]
        ax.set_varlabels(class_labels, fontfamily="Hiragino Sans GB")

        # make the axis lightgrey
        ax.spines['polar'].set_color('#696969')
        # make joins in the polygon round with large radius
        ax.spines['polar'].set_joinstyle('round')
        ax.spines['polar'].set_linewidth(1)

        # make gridlines lightgrey and dotted
        ax.yaxis.grid(color='#A9A9A9')
        ax.yaxis.grid(linestyle='dotted')



    if show:
        plt.show()
        return None
    return fig, ax


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # 1. Define your classes
    classes = ["Berlin", "Vienna", "London", "Munich"]
    
    # 2. Your raw probability data (N_samples x N_classes)
    raw_data = np.array([
        [0.9,  0.1,  0.0,  0.0],
        [0.8,  0.15, 0.05, 0.0],
        [0.75, 0.15, 0.08, 0.02]
    ])
    
    # 3. Calculate Credal Bounds (Min/Max)
    min_values = raw_data.min(axis=0).tolist()
    max_values = raw_data.max(axis=0).tolist()

    # 4. Structure the data dictionary (MLE and Ground-Truth removed)
    plot_data = {
        "City Classification Probability Intervals": {
            "Credal set": {
                "start": min_values,
                "end": max_values,
            }
        }
    }

    # 5. Define Colors and Markers (MLE and Ground-Truth removed)
    plot_colors = {
        "Credal set": BLUE.get_hex()
    }
    
    plot_markers = {
        "Credal set": None
    }

    # 6. Call the plotting function
    fig, ax = spider_plot(
        data=plot_data,
        class_labels=classes,
        colors=plot_colors,
        markers=plot_markers,
        show=False,             
        interval_thickness=0.1, 
        fig_size=(9, 9),
        ylim=(0.0,1.0)
    )

    # Move legend outside the plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # 7. Save and Show
    filename = "city_classification_intervals.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Plot saved successfully as: {os.path.abspath(filename)}")
    
    plt.show()
