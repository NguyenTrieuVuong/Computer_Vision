function openCvReady(){
    cv["onRuntimeInitialized"]=()=>{
        console.log("OpenCV Ready")
        // read an image from the image source and convert to opencv format
        let imgMain = cv.imread("img-main"); // image in opencv mat format
        cv.imshow("main-canvas", imgMain); //to free the memory allocated
        imgMain.delete();

        // ___________RGB Button____________
        document.getElementById("button-rgb").onclick = function(){
            let imgMain = cv.imread("img-main");
            cv.imshow("main-canvas", imgMain); //to free the memory allocated
            imgMain.delete();
        };

        // ___________Gray Button____________
        document.getElementById("button-gray").onclick = function(){
            let imgMain = cv.imread("img-main");
            let imgGray = imgMain.clone();
            // Converting image to grayscale
            cv.cvtColor(imgMain, imgGray, cv.COLOR_RGBA2GRAY, 0)
            cv.imshow("main-canvas", imgGray);
            imgMain.delete(); 
            imgGray.delete();
        };

        // ___________Blur Button____________
        document.getElementById("button-blur").onclick = function(){
            let imgMain = cv.imread("img-main");
            let imgBlur = imgMain.clone();
            // Adding Blur
            let ksize = new cv.Size(49,49)
            cv.GaussianBlur(imgMain, imgBlur, ksize, 0)
            cv.imshow("main-canvas", imgBlur);
            imgMain.delete(); 
            imgBlur.delete();
        };

        // ___________Edge Button____________
        document.getElementById("button-edge").onclick = function(){
            let imgMain = cv.imread("img-main");
            let imgCanny = imgMain.clone();
            // Find Edges using canny edge detector
            cv.Canny(imgMain, imgCanny, 50, 100)
            cv.imshow("main-canvas", imgCanny);
            imgMain.delete(); 
            imgCanny.delete();
        };

    }
}