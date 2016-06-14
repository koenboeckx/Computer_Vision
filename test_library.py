from project_library import *

    
if __name__ == '__main__': 
    landmarks = load_landmarks(type='2')
#    ## Prepare image
    filename = './Data/Radiographs/01.tif'
    idx = int(filename[-5])
    # get part of image    
    left = 1000; top = 500; right = 2000; lower = 1500;
    image = Image.open(filename).convert('L')
    image = image.crop((left, top, right, lower))
        
    G.root = tk.Tk()
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(image=photo)
    label.image = photo # keep a reference!
    label.pack()
    
    G.root.bind('<1>', mouse_down)
    G.root.mainloop()
    
    G.best_init[1] = (G.trans_x, G.trans_y)

    image = np.array(image)
    
#    ## image cleanup
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)    
    
    weights = {'t' : 1.0,
               'theta' : 2.0,
               's' : 1.0,
               'b' : np.ones((80, 1))}
    weights['b'][1] = 1.0
    weights['b'][2] = 1.0
#    
    X = adjust_shape(image, landmarks, centrum=G.best_init[idx], n_iters=25,
                     weights=weights, method=1)
    show_image(image, X); plt.hold(True)
    landmarks = load_landmarks(type='2')
    plot_shape([landmarks[0, :].reshape((40, 2)) - np.array([left, top])])
    plt.legend(['solution', 'original'])
    plt.axis('tight')

#    results = find_all(image, start_point=G.best_init[1],
#                       weights=weights, method=2)
