# -*- coding: utf-8 -*-
"""
Created on Mon May 27 08:25:44 2024

@author: champ
"""


from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.colors as pltcolors

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import  QFileDialog
from matplotlib.widgets import RadioButtons, TextBox, Button, RectangleSelector, PolygonSelector

import time

#%%
def ContourMultiPlot(XData2D, YData2D, ZData2D, # must be in meshgrid format
                          OverlayScatter_red = None, # 
                          OverlayScatter_red_labels = None,
                          OverlayScatter_red_name = 'red series',
                          OverlayScatter_red_rotation = 45,
                          
                          x_axis_label = 'X',
                          y_axis_label = 'Y',
                          z_axis_label = 'Z',
                          
                          num_contours = 25,
                          
                          Title = '',
                          aspect_style = 'equal'):
    
    fig, ax = plt.subplots()
    
    CS = ax.contourf(XData2D, YData2D, ZData2D, num_contours, cmap = 'viridis', zorder = -1)
    
    CSB = ax.contour(XData2D, YData2D, ZData2D, num_contours, colors='k', zorder = 6)
    
    ax.clabel(CSB, inline=1, fontsize=10)
    
    if type(OverlayScatter_red) != type(None):
        sensors_scatter_plot = plt.scatter(OverlayScatter_red[:,0], 
                                           OverlayScatter_red[:,1], 
                                           c = 'red',
                                           s = 50,
                                           marker = 'o',
                                           label = OverlayScatter_red_name,
                                           zorder = 8)
        if type(OverlayScatter_red_labels) != type(None):
            
            for idl, data_label in enumerate(OverlayScatter_red_labels):
                pos_x, pos_y = OverlayScatter_red[idl,:2]
                plt.text(pos_x, pos_y, 
                         '  '+str(data_label), 
                         fontsize = 8, 
                         rotation=OverlayScatter_red_rotation, 
                         c = 'red',
                         clip_on = True)        

    ax.set_aspect(aspect_style)    
    
    ax.set_title(Title)
    
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    leg = plt.legend()   
    leg.set_zorder(20)
    fig.colorbar(CS)
    
    plt.show()


#%%
def PlotTriSurface(XDat, YDat, ZDat, Title = '', x_label = '', y_label = '', z_label = ''):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    pos = ax.plot_trisurf(XDat, YDat, ZDat, linewidth=0.2, antialiased=True, cmap=plt.cm.viridis)

    ax.set_title(Title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    fig.colorbar(pos, ax=ax)
    plt.show()   


#%%
def PlotSurfaceGrid(XX, YY, ZZ, 
                    x_label = '', 
                    y_label = '', 
                    z_label = '', 
                    Title = '', 
                    max_color = None, 
                    min_color = None, 
                    cmap_to_use = plt.cm.gist_earth,
                    x_range = [],
                    y_range = [],
                    z_range = []):   
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(12,9))
    ls = LightSource(270, 45)
    ZZZ = deepcopy(ZZ)
    if type(max_color) == type(None):
        pass
    else:
        ZZZ[ZZZ > max_color] = max_color
    if type(min_color) == type(None):
        pass
    else:
        ZZZ[ZZZ < min_color] = min_color

    try:    
        rgb = ls.shade(ZZZ, plt.get_cmap('gist_earth'), vert_exag = 0.1, blend_mode = 'soft', vmin = min_color, vmax = max_color)
    except:
        rgb = ls.shade(ZZZ, plt.get_cmap('gist_earth'), vert_exag = 0.1, blend_mode = 'soft')

    
    surf = ax.plot_surface(XX, YY, ZZ , rstride = 1, cstride = 1, facecolors = rgb, linewidth = 0, antialiased=False, shade = False)
    
    ax.set_title(Title)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    m = cm.ScalarMappable(cmap=plt.cm.gist_earth)
    try:
        m.set_array(np.linspace(min_color, max_color, num = 100))
    except:
        m.set_array(ZZ)

    
    fig.colorbar(m, ax=ax)

    if len(x_range):
        ax.set_xlim(x_range[0], x_range[1])
    if len(y_range):
        ax.set_ylim(y_range[0], y_range[1])
    if len(z_range):
        ax.set_zlim(z_range[0], z_range[1]) 

    plt.show()
        
        
#%%

def ShowImage(X, 
              colormap = plt.cm.viridis, 
              min_color_val = None, 
              max_color_val = None, 
              overlay_x = [], 
              overlay_y = [], 
              x_labels = [], 
              y_labels = [], 
              overlay_label='', 
              LogScale = False,
              Title = '',
              save_fig_filepath = None):
    
     
    fig, ax = plt.subplots(figsize=(16,9))
    if type(min_color_val) == type(None):
        VMIN = X.min()
    else:
        VMIN = min_color_val
    if type(max_color_val) == type(None):
        VMAX = X.max()
    else:
        VMAX = max_color_val
        
    if LogScale:
        colornorm = pltcolors.LogNorm(vmin=VMIN, vmax=VMAX)
        pos = ax.imshow(X, interpolation='nearest', norm=colornorm, cmap = colormap)
    else:
        colornorm = None
        pos = ax.imshow(X, interpolation='nearest', norm=colornorm, cmap = colormap,vmin = VMIN, vmax =VMAX)

    numrows, numcols = X.shape
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = X[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)
    
    ax.format_coord = format_coord
    fig.colorbar(pos, ax=ax)

    ax.set_title(Title)

    plt.show()



#%%


def ShowImageRGB(imageRGB, Title = '', InsetTitle = '', dark_mode = False):
    
    if dark_mode:
        fig, ax = plt.subplots(facecolor='black', figsize=(16,9))
        fig.canvas.toolbar_visible = False
        text_color = 'red'
        extra_text = '\n[f]:fullscreen, [o]:zoom, [ctrl+w]:close, [p]:pan, [h]:reset'
        #import matplotlib
        #matplotlib.rcParams['toolbar'] = 'None'
        
        # toolbar = plt.get_current_fig_manager().toolbar
        # unwanted_buttons = ['Subplots','Save', 'Zoom']
        # print('ggg', toolbar, dir(toolbar))
        
        # for x in toolbar.actions():
        #     if x.text() in unwanted_buttons:
        #         toolbar.removeAction(x)
        
        
    else:
        
        fig, ax = plt.subplots(figsize=(16,9))
        fig.canvas.toolbar_visible = True
        text_color = 'white'
        extra_text = ''
    
    if len(Title) > 0:
        ax.set_title(Title)
    if len(InsetTitle) > 0:
        middle_coord_x = imageRGB.shape[1]/2
        top_coord_y = 0.005 * imageRGB.shape[0]
        plt.text(middle_coord_x, top_coord_y, InsetTitle+extra_text, fontsize=8,  ha='center',
         va='top', wrap=True, color=text_color)

    
        ax.set_title(Title)
    
    plt.imshow(imageRGB)
    #fig.patch.set_visible(False)
    
    ax.axis('off')
    #plt.tight_layout(pad = 1.00)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.show()

#%%

def QuickImage(ImageRGB, plow = 5, phigh = 95, LogScale = False):
    try:
        Lum = ImageRGB.mean(axis = 2)
    except:
        Lum = ImageRGB
    
    plowVal, phighVal = np.percentile(Lum, [plow, phigh])
    
    ShowImage(Lum, min_color_val=plowVal, max_color_val=phighVal, LogScale = LogScale)
    
#%%

def QuickInspectRGB(raw_RGB_frame, perc_black=5, perc_white = 95, Title = ''):
    
    _RGB = deepcopy(raw_RGB_frame)
    
    
    R_low, R_high = np.percentile(_RGB[:,:,0], [perc_black, perc_white])
    G_low, G_high = np.percentile(_RGB[:,:,1], [perc_black, perc_white])
    B_low, B_high = np.percentile(_RGB[:,:,2], [perc_black, perc_white])
    
    _RGB -= np.array([[[R_low, G_low, B_low]]])
    _RGB /= np.array([[[R_high-R_low, G_high-G_low, B_high-B_low]]])
    
    r_text = 'Red:   black = ' + str(int(round(R_low*2**16))) + ' , white = ' + str(int(round(R_high*2**16))) + ' , range = ' + str(int(round((R_high-R_low)*2**16))) + ' (counts)' 
    g_text = 'Green: black = ' + str(int(round(G_low*2**16))) + ' , white = ' + str(int(round(G_high*2**16))) + ' , range = ' + str(int(round((G_high-G_low)*2**16))) + ' (counts)' 
    b_text = 'Blue:  black = ' + str(int(round(B_low*2**16))) + ' , white = ' + str(int(round(B_high*2**16))) + ' , range = ' + str(int(round((B_high-B_low)*2**16))) + ' (counts)' 
    
    
    ShowImageRGB(_RGB, InsetTitle = r_text + '\n' + g_text + '\n' + b_text, Title = Title)
    
    

#%%






def CollectPointsImage(InputImage, 
                       min_color_val = None,
                       max_color_val = None,
                       message = '',
                       colormap = plt.cm.viridis,
                       linewidth = 0):

    if len(InputImage.shape) == 3:
        mode = 'RGB'
    else:
        mode = 'Gray'
    
    
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_title('Select points with the mouse... ' + message)
    

    if type(min_color_val) == type(None):
        VMIN = InputImage.min()
    else:
        VMIN = min_color_val
    if type(max_color_val) == type(None):
        VMAX = InputImage.max()
    else:
        VMAX = max_color_val
    
    if mode == 'Gray':
        pos = ax.imshow(InputImage, interpolation='nearest', cmap = colormap, vmin = VMIN, vmax =VMAX)
        
    else:
        plt.imshow(InputImage)
        
    fig.show()
    
    selector = PolygonSelector(ax, 
                               lambda *args: None,
                               useblit = True,
                               props = dict(color='magenta', linestyle='-', linewidth=linewidth, alpha=0.5, markersize = 50)) #
    # useblit = True
    
    print('made it here')
    #selector.disconnect_events()
    print('made it here too')
    
    
    return selector, selector.verts
    











    
    
#%%

def MultiScatterPlot(ListX, 
                     ListY, 
                     Colors = [], 
                     Sizes = [], 
                     Labels = [], 
                     Lines = False, 
                     alpha = 0.8, 
                     LineWidth = 3, 
                     Title = '', 
                     XLabel = '', 
                     YLabel = '',
                     darkness = False,
                     primary_cmap = plt.cm.viridis,
                     alt_cmap = plt.cm.plasma,
                     CollectROI = False,
                     save_fig_filepath = None,
                     y_lim = None,
                     x_lim = None):

    primary_low_clip = 0.1
    primary_high_clip = .9        
        
    if len(Colors) ==0:
        Colors = primary_cmap(np.linspace(primary_low_clip, primary_high_clip, num = len(ListX)))
    if len(Sizes) == 0:
        Sizes = [20]*len(ListX)
    if len(Labels)==0:
        Labels = ['series ' + str(ii) for ii in range(len(ListX))]

    if darkness:
        plt.style.use('dark_background')
        
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title(Title)

    if darkness:
        ax.set_facecolor('dimgray')
    for ii in range(len(ListX)):
        if Lines:
            ax.plot(ListX[ii],ListY[ii], 'o-', ms=0.2*Sizes[ii], c = Colors[ii], zorder=2*ii, lw=LineWidth, label = Labels[ii], alpha = alpha)
        else:
            try:
                ax.scatter(ListX[ii],ListY[ii], c = Colors[ii].reshape(1,4), s = Sizes[ii], label = Labels[ii], lw = 2, zorder = 2*ii+1, alpha = alpha)        
            except:
                ax.scatter(ListX[ii],ListY[ii], c = Colors[ii], s = Sizes[ii], label = Labels[ii], lw = 2, zorder = 2*ii+1, alpha = alpha)        


    ax.set_xlabel(XLabel)
    ax.set_ylabel(YLabel)
    
    min_x = min([zz.min() for zz in ListX])
    max_x = max([zz.max() for zz in ListX])
    
    min_y = min([zz.min() for zz in ListY])
    max_y = max([zz.max() for zz in ListY])
    
    y_span_buffer = 0.05 * (max_y - min_y)
    x_span_buffer = 0.05 * (max_x - min_x)
    
    if y_span_buffer == 0:
        y_span_buffer = 1.0
    if x_span_buffer == 0:
        x_span_buffer = 1.0
    
    if type(x_lim) == type(None):
        ax.set_xlim((min_x - x_span_buffer, max_x + x_span_buffer))
    else:
        ax.set_xlim(x_lim)
        
    if type(y_lim) == type(None):
        ax.set_ylim((min_y - y_span_buffer, max_y + y_span_buffer))
    else:
        ax.set_ylim(y_lim)
    
    plt.legend(prop={"size":8.5})
     
    plt.show()

#%%
def PlotHistInt(imageInt, domain = 'int'):
    
    counts_I = np.bincount(np.array(np.round(imageInt.ravel()*2**16), dtype = int))
    bins_I = np.arange(len(counts_I))
    
    if domain == 'int':
        MultiScatterPlot([bins_I], [counts_I], Colors = ['k'], Labels = ['Intensity'])
    else:
        MultiScatterPlot([bins_I/2**16], [counts_I], Colors = ['k'], Labels = ['Intensity'])
    return bins_I, counts_I

#%%

def PlotHistRGB(imageRGB, domain = 'int', return_counts = False, skipplot = False, mask = None):
    
    if type(mask) == type(None):
        
        nnIntRGB = np.array(np.round(imageRGB*2**16), dtype = int)
        nnIntRGB[nnIntRGB < 0] = 0
        
        counts_R = np.bincount(nnIntRGB[:,:,0].ravel())
        counts_G = np.bincount(nnIntRGB[:,:,1].ravel())
        counts_B = np.bincount(nnIntRGB[:,:,2].ravel())
        
        # counts_R = np.bincount(np.array(np.round(imageRGB[:,:,0].ravel()*2**16), dtype = int))
        # counts_G = np.bincount(np.array(np.round(imageRGB[:,:,1].ravel()*2**16), dtype = int))
        # counts_B = np.bincount(np.array(np.round(imageRGB[:,:,2].ravel()*2**16), dtype = int))
    else:
        counts_R = np.bincount(np.array(np.round(imageRGB[:,:,0][mask]*2**16), dtype = int))
        counts_G = np.bincount(np.array(np.round(imageRGB[:,:,1][mask]*2**16), dtype = int))
        counts_B = np.bincount(np.array(np.round(imageRGB[:,:,2][mask]*2**16), dtype = int))
        
        
        
    bins_R = np.arange(len(counts_R))
    bins_G = np.arange(len(counts_G))
    bins_B = np.arange(len(counts_B))
    
    if not skipplot:
        if domain == 'int':
            MultiScatterPlot([bins_R, bins_G, bins_B], [counts_R, counts_G, counts_B], Colors = ['r', 'g', 'b'], Labels = ['Red', 'Green', 'Blue'])
        else:
            MultiScatterPlot([bins_R/2**16, bins_G/2**16, bins_B/2**16], [counts_R, counts_G, counts_B], Colors = ['r', 'g', 'b'], Labels = ['Red', 'Green', 'Blue'])

    if return_counts:
        C_RGB = np.zeros((2**16,3), dtype = int)
        
        C_RGB[:min(2**16,len(counts_R)), 0] = counts_R[:min(2**16,len(counts_R))]
        C_RGB[:min(2**16,len(counts_G)), 1] = counts_G[:min(2**16,len(counts_G))]
        C_RGB[:min(2**16,len(counts_B)), 2] = counts_B[:min(2**16,len(counts_B))]

        return C_RGB





#%%

def GetColors(n, cmap_to_use = plt.cm.viridis):
    Colors = cmap_to_use(np.linspace(.1, .9, num = n))
    return Colors        
        
#%%

def MultiScatterPlot3D(ListX, ListY, ListZ, 
                       Colors = [], 
                       Sizes = [], 
                       Labels = [], 
                       Title = '',
                       XLabel = 'X',
                       YLabel = 'Y',
                       ZLabel = 'Z',
                       aspect = 'garbage',
                       alpha = 0.8,
                       darkness = False):
    
    if len(Labels) == 0:
        Labels = ['series ' + str(ijk) for ijk in range(len(ListX))]

    if len(Sizes) == 0:
        Sizes = [20]*len(ListX)
        
    if len(Colors)==0:
        Colors = plt.cm.viridis(np.linspace(.1, .9, num = len(ListX)))
        
    if darkness:
        plt.style.use('dark_background')
    
    
    print('debug:', Colors, Colors[0])
    fig = plt.figure(figsize=(9, 9), dpi=70)
    ax = fig.add_subplot(111, projection = '3d')

    if darkness and False:
        ax.w_xaxis.pane.set_alpha(0.25)
        ax.w_yaxis.pane.set_alpha(0.25)
        ax.w_zaxis.pane.set_alpha(0.25)
    
    for series_idx in range(len(ListX)):
        ax.scatter(ListX[series_idx], ListY[series_idx], ListZ[series_idx],
                   marker='o',
                   c = Colors[series_idx],
                   s = Sizes[series_idx],
                   label = Labels[series_idx],
                   alpha = alpha)
        
    ax.set_title(Title)
    ax.set_xlabel(XLabel)
    ax.set_ylabel(YLabel)
    ax.set_zlabel(ZLabel)

    if aspect == 'equal':
        
        all_x = np.concatenate(ListX) 
        all_y = np.concatenate(ListY) 
        all_z = np.concatenate(ListZ) 

        max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
        
        mid_x = (all_x.max()+all_x.min()) * 0.5
        mid_y = (all_y.max()+all_y.min()) * 0.5
        mid_z = (all_z.max()+all_z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)        
        
    plt.legend()
    plt.show()




#%%

def CollectUserInput(ListOfInteractiveComponentTypes,  # list of either 'RadioButtons','TextBox', 'FileDialog', 'DirectoryDialog'
                     ListOfInteractiveComponentLabels, # unique names for these elements
                     ListOfComponentPayloads,          # for RadioButtons, these are the list of button values, for TextBox, this is the initial fill value
                     v_space_per_element = 0.08 ,
                     left_text_space = 0.2):

    fig, ax = plt.subplots(figsize=(13, 13))
    ax.axis('off')
    plt.subplots_adjust(left=0.8)
    interactive_objects = []
    
    total_elements = sum([len(payload) if ListOfInteractiveComponentTypes[ipayload]=='RadioButtons' else 1 for ipayload, payload in enumerate(ListOfComponentPayloads)])
    
    element_positions = 1.0 - v_space_per_element * np.arange(total_elements+1)
    
    element_index = 1
    
    class FileHolder():
        def __init__(self, caption, parent_axis):
            self.complete = False
            self.filePath = ''
            self.caption = caption
            self.ax = parent_axis
            self.dialog_open = False
        def fetch_file(self, event):
            self.dialog_open = True
            filePath, _trash = QFileDialog.getOpenFileName(None,caption=self.caption)
            self.dialog_open = False
            text_kwargs = dict(ha='left', va='center', fontsize=10, color='C1')
            self.ax.text(0.05, 0.5, filePath, **text_kwargs)
            self.filePath = filePath
        def fetch_dir(self, event):
            self.dialog_open = True
            filePath = str(QFileDialog.getExistingDirectory(None, self.caption)) 
            self.dialog_open = False
            text_kwargs = dict(ha='left', va='center', fontsize=10, color='C1')
            self.ax.text(0.05, 0.5, filePath, **text_kwargs)
            self.filePath = filePath
            
    class Switch():
        def __init__(self):
            self.complete = False
        def finalize(self, event):
            self.complete = True

    for iComp, Comp in enumerate(ListOfInteractiveComponentTypes):
        
        if Comp == 'FileDialog':
            default = ListOfComponentPayloads[iComp]
            _label = ListOfInteractiveComponentLabels[iComp]
            axcolor = 'lavender'
            
            #[left, bottom, width, height]
            _fax = plt.axes([left_text_space, element_positions[element_index], .2, 0.9*v_space_per_element], 
                            facecolor=axcolor,
                            frameon = False,
                            )

            _fax.axes.xaxis.set_visible(False)
            _fax.axes.yaxis.set_visible(False)
            
            file_holder_object = FileHolder(_label, _fax)
            axfilebutton = plt.axes([0.02, element_positions[element_index], 0.16, v_space_per_element])
            bfiledialog = Button(axfilebutton, _label)
            bfiledialog.on_clicked(file_holder_object.fetch_file)    
        
            interactive_objects.append([_label, Comp, bfiledialog, file_holder_object])
            element_index += 1
        
        
        
        elif Comp == 'DirectoryDialog':
            default = ListOfComponentPayloads[iComp]
            _label = ListOfInteractiveComponentLabels[iComp]
            axcolor = 'lightgoldenrodyellow'
            
            #[left, bottom, width, height]
            _fax = plt.axes([left_text_space, element_positions[element_index], .2, 0.9*v_space_per_element], 
                            facecolor=axcolor,
                            frameon = False,
                            )

            _fax.axes.xaxis.set_visible(False)
            _fax.axes.yaxis.set_visible(False)
            
            file_holder_object = FileHolder(_label, _fax)
            axfilebutton = plt.axes([0.02, element_positions[element_index], 0.16, v_space_per_element])
            bfiledialog = Button(axfilebutton, _label)
            bfiledialog.on_clicked(file_holder_object.fetch_dir)    
        
            interactive_objects.append([_label, Comp, bfiledialog, file_holder_object])
            element_index += 1
        
    
        elif Comp == 'RadioButtons':
            RB = ListOfComponentPayloads[iComp]
            axcolor = 'lightgoldenrodyellow'
            #[left, bottom, width, height]
            _label = ListOfInteractiveComponentLabels[iComp]
            _rax = plt.axes([.02, element_positions[element_index + len(RB)-1], .3, len(RB)*v_space_per_element], 
                            facecolor=axcolor,
                            frameon = False,
                            ) #aspect = 'equal')
            ax.set_ylabel(_label)
            _radio = RadioButtons(_rax, tuple([_label + " " + str(rb) for rb in RB]))

            interactive_objects.append([_label, Comp, '', _radio])
            element_index += len(RB)
        

        elif Comp == 'TextBox':
            #[left, bottom, width, height]
            _label = ListOfInteractiveComponentLabels[iComp]
            _axbox = fig.add_axes([left_text_space, element_positions[element_index], .6, v_space_per_element])
            
            _text_box = TextBox(_axbox, _label)
            _text_box.set_val(ListOfComponentPayloads[iComp])  
            
            interactive_objects.append([_label, Comp, '', _text_box])
            
            element_index += 1
        
        else:
            print('unsupported component type:', Comp)
            

        
    finalize_switch = Switch()
    axfinalize = plt.axes([0.7, 0.01, 0.25, 0.075])
    bfinalize = Button(axfinalize, 'Proceed to next step')
    bfinalize.on_clicked(finalize_switch.finalize)
            
    plt.show() 

    while not finalize_switch.complete:
        
        fig.canvas.draw() 
        fig.canvas.flush_events()
        time.sleep(0.01)

    collected_user_data = []
    
    for widget_index, widget in enumerate(interactive_objects):
        
        _label, _type, _trash, _object = widget
        
        if _type == 'FileDialog':
            _collected_data = deepcopy(_object.filePath)
        elif _type == 'DirectoryDialog':
            _collected_data = deepcopy(_object.filePath)
        elif _type == 'RadioButtons':
            _collected_data = deepcopy(_object.value_selected).replace(_label + " ", '')
        elif _type == 'TextBox': 
            _collected_data = deepcopy(_object.text)
            
        try: 
            _collected_data = int(_collected_data)
        except:
            try:
                _collected_data = float(_collected_data)
            except:
                _collected_data = str(_collected_data)
                    
        collected_user_data.append([_label,_type, _collected_data ])
        
    plt.close(fig = fig)
    return collected_user_data            
            






