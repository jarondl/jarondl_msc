#### Copied from  https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/figure.py
####    and from  https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/tight_layout.py
#### to backport tight_layout. 


def tight_layout(figure, renderer=None, pad=1.2, h_pad=None, w_pad=None):
        """  Copied from : https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/figure.py
Adjust subplot parameters to give specified padding.

Parameters:

pad : float
padding between the figure edge and the edges of subplots, as a fraction of the font-size.
h_pad, w_pad : float
padding (height/width) between edges of adjacent subplots.
Defaults to `pad_inches`.
"""

        #from tight_layout import auto_adjust_subplotpars, get_renderer

        if renderer is None:
            renderer = get_renderer(figure)

        subplotspec_list = []
        subplot_list = []
        nrows_list = []
        ncols_list = []
        ax_bbox_list = []

        subplot_dict = {} # for axes_grid1, multiple axes can share
                          # same subplot_interface. Thus we need to
                          # join them together.

        for ax in figure.axes:
            locator = ax.get_axes_locator()
            if hasattr(locator, "get_subplotspec"):
                subplotspec = locator.get_subplotspec().get_topmost_subplotspec()
            elif hasattr(ax, "get_subplotspec"):
                #subplotspec = ax.get_subplotspec().get_topmost_subplotspec()
                subplotspec = ax.get_subplotspec() # .get_topmost_subplotspec()
            else:
                continue

#            if (subplotspec is None) or \
#                   subplotspec.get_gridspec().locally_modified_subplot_params():
#                continue

            subplots = subplot_dict.setdefault(subplotspec, [])

            if not subplots:
                myrows, mycols, _, _ = subplotspec.get_geometry()
                nrows_list.append(myrows)
                ncols_list.append(mycols)
                subplotspec_list.append(subplotspec)
                subplot_list.append(subplots)
                ax_bbox_list.append(subplotspec.get_position(figure))

            subplots.append(ax)

        max_nrows = max(nrows_list)
        max_ncols = max(ncols_list)

        num1num2_list = []
        for subplotspec in subplotspec_list:
            rows, cols, num1, num2 = subplotspec.get_geometry()
            div_row, mod_row = divmod(max_nrows, rows)
            div_col, mod_col = divmod(max_ncols, cols)
            if (mod_row != 0) or (mod_col != 0):
                raise RuntimeError("")

            rowNum1, colNum1 = divmod(num1, cols)
            if num2 is None:
                rowNum2, colNum2 = rowNum1, colNum1
            else:
                rowNum2, colNum2 = divmod(num2, cols)

            num1num2_list.append((rowNum1*div_row*max_ncols + colNum1*div_col,
                                  ((rowNum2+1)*div_row-1)*max_ncols + (colNum2+1)*div_col-1))


        kwargs = auto_adjust_subplotpars(figure, renderer,
                                         nrows_ncols=(max_nrows, max_ncols),
                                         num1num2_list=num1num2_list,
                                         subplot_list=subplot_list,
                                         ax_bbox_list=ax_bbox_list,
                                         pad=pad, h_pad=h_pad, w_pad=w_pad)

        figure.subplots_adjust(**kwargs)


import matplotlib
from matplotlib.transforms import TransformedBbox, Bbox

from matplotlib.font_manager import FontProperties
rcParams = matplotlib.rcParams



def _get_left(tight_bbox, axes_bbox):
    return axes_bbox.xmin - tight_bbox.xmin

def _get_right(tight_bbox, axes_bbox):
    return tight_bbox.xmax - axes_bbox.xmax

def _get_bottom(tight_bbox, axes_bbox):
    return axes_bbox.ymin - tight_bbox.ymin

def _get_top(tight_bbox, axes_bbox):
    return tight_bbox.ymax - axes_bbox.ymax


def auto_adjust_subplotpars(fig, renderer,
                            nrows_ncols,
                            num1num2_list,
                            subplot_list,
                            ax_bbox_list=None,
                            pad=1.2, h_pad=None, w_pad=None,
                            rect=None):
    """
Return a dictionary of subplot parameters so that spacing between
subplots are adjusted. Note that this function ignore geometry
information of subplot itself, but uses what is given by
*nrows_ncols* and *num1num2_list* parameteres. Also, the results could be
incorrect if some subplots have ``adjustable=datalim``.
Parameters:

nrows_ncols
number of rows and number of columns of the grid.

num1num2_list
list of numbers specifying the area occupied by the subplot

subplot_list
list of subplots that will be used to calcuate optimal subplot_params.

pad : float
padding between the figure edge and the edges of subplots, as a fraction of the font-size.
h_pad, w_pad : float
padding (height/width) between edges of adjacent subplots.
Defaults to `pad_inches`.

rect
[left, bottom, right, top] in normalized (0, 1) figure coordinates.
"""


    rows, cols = nrows_ncols

    pad_inches = pad * FontProperties(size=rcParams["font.size"]).get_size_in_points() / renderer.dpi

    if h_pad is not None:
        vpad_inches = h_pad * FontProperties(size=rcParams["font.size"]).get_size_in_points() / renderer.dpi
    else:
        vpad_inches = pad_inches

    if w_pad is not None:
        hpad_inches = w_pad * FontProperties(size=rcParams["font.size"]).get_size_in_points() / renderer.dpi
    else:
        hpad_inches = pad_inches
        

    if len(subplot_list) == 0:
        raise RuntimeError("")

    if len(num1num2_list) != len(subplot_list):
        raise RuntimeError("")

    if rect is None:
        margin_left, margin_bottom, margin_right, margin_top = None, None, None, None
    else:
        margin_left, margin_bottom, _right, _top = rect
        if _right: margin_right = 1. - _right
        else: margin_right = None
        if _top: margin_top = 1. - _top
        else: margin_top = None
        
    vspaces = [[] for i in range((rows+1)*cols)]
    hspaces = [[] for i in range(rows*(cols+1))]
    
    union = Bbox.union

    if ax_bbox_list is None:
        ax_bbox_list = []
        for subplots in subplot_list:
            ax_bbox = union([ax.get_position(original=True) for ax in subplots])
            ax_bbox_list.append(ax_bbox)
        
    for subplots, ax_bbox, (num1, num2) in zip(subplot_list,
                                               ax_bbox_list,
                                               num1num2_list):

        #ax_bbox = union([ax.get_position(original=True) for ax in subplots])

        tight_bbox_raw = union([ax.get_tightbbox(renderer) for ax in subplots])
        tight_bbox = TransformedBbox(tight_bbox_raw, fig.transFigure.inverted())

        row1, col1 = divmod(num1, cols)


        if num2 is None:
            # left
            hspaces[row1 * (cols+1) + col1].append(_get_left(tight_bbox, ax_bbox))
            # right
            hspaces[row1 * (cols+1) + (col1+1)].append(_get_right(tight_bbox, ax_bbox))
            # top
            vspaces[row1 * cols + col1].append(_get_top(tight_bbox, ax_bbox))
            # bottom
            vspaces[(row1+1) * cols + col1].append(_get_bottom(tight_bbox, ax_bbox))

        else:
            row2, col2 = divmod(num2, cols)
        
            for row_i in range(row1, row2+1):
                # left
                hspaces[row_i * (cols+1) + col1].append(_get_left(tight_bbox, ax_bbox))
                # right
                hspaces[row_i * (cols+1) + (col2+1)].append(_get_right(tight_bbox, ax_bbox))
            for col_i in range(col1, col2+1):
                # top
                vspaces[row1 * cols + col_i].append(_get_top(tight_bbox, ax_bbox))
                # bottom
                vspaces[(row2+1) * cols + col_i].append(_get_bottom(tight_bbox, ax_bbox))


    fig_width_inch, fig_height_inch = fig.get_size_inches()

    # margins can be negative for axes with aspect applied. And we
    # append + [0] to make minimum margins 0

    if not margin_left:
        margin_left = max([sum(s) for s in hspaces[::cols+1]] + [0])
        margin_left += pad_inches / fig_width_inch
    
    if not margin_right:
        margin_right = max([sum(s) for s in hspaces[cols::cols+1]] + [0])
        margin_right += pad_inches / fig_width_inch

    if not margin_top:
        margin_top = max([sum(s) for s in vspaces[:cols]] + [0])
        margin_top += pad_inches / fig_height_inch

    if not margin_bottom:
        margin_bottom = max([sum(s) for s in vspaces[-cols:]] + [0])
        margin_bottom += pad_inches / fig_height_inch
    

    kwargs = dict(left=margin_left,
                  right=1-margin_right,
                  bottom=margin_bottom,
                  top=1-margin_top)

    if cols > 1:
        hspace = max([sum(s) for i in range(rows) for s in hspaces[i*(cols+1)+1:(i+1)*(cols+1)-1]])
        hspace += hpad_inches / fig_width_inch
        h_axes = ((1-margin_right-margin_left) - hspace * (cols - 1))/cols

        kwargs["wspace"]=hspace/h_axes

    if rows > 1:
        vspace = max([sum(s) for s in vspaces[cols:-cols]])
        vspace += vpad_inches / fig_height_inch
        v_axes = ((1-margin_top-margin_bottom) - vspace * (rows - 1))/rows

        kwargs["hspace"]=vspace/v_axes

    return kwargs
    

def get_renderer(fig):
    if fig._cachedRenderer:
        renderer = fig._cachedRenderer
    else:
        canvas = fig.canvas

        if canvas and hasattr(canvas, "get_renderer"):
            renderer = canvas.get_renderer()
        else:
            # not sure if this can happen
            print "tight_layout : falling back to Agg renderer"
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            canvas = FigureCanvasAgg(fig)
            renderer = canvas.get_renderer()

    return renderer
