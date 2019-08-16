
const getHeadpose = async function(image,landMarks2d,verbose=false){


    window.beforeunload = () => {
        im.delete();
        imagePoints.delete();
        distCoeffs.delete();
        rvec.delete();
        tvec.delete();
        pointZ.delete();
        pointY.delete();
        pointX.delete();
        noseEndPoint2DZ.delete();
        nose_end_point2DY.delete();
        nose_end_point2DX.delete();
        jaco.delete();
      };

    const width = image.cols;
    const height = image.rows;
    
    const lm2DIndex =  [33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]

    const modelPoints = cv.Mat.zeros(15, 3, cv.CV_64FC1);

    let Points3D = [ 0.000000,  0.000000,  6.763430,  
             6.825897,  6.760612,  4.402142,   
             1.330353,  7.122144,  6.903745,   
            -1.330353,  7.122144,  6.903745,  
            -6.825897,  6.760612,  4.402142,   
             5.311432,  5.485328,  3.987654,  
             1.789930,  5.393625,  4.413414,   
            -1.789930,  5.393625,  4.413414,   
            -5.311432,  5.485328,  3.987654,   
             2.005628,  1.409845,  6.165652,  
            -2.005628,  1.409845,  6.165652,  
             2.774015, -2.080775,  5.048531,   
            -2.774015, -2.080775,  5.048531,   
             0.000000, -3.116408,  6.097667,   
             0.000000, -7.415691,  4.070434 ]    

    
    const Points3Dnew = [];

    for(i=0;i<Points3D.length;i++){
        Points3Dnew.push(Points3D[i]*40);
    }

    console.log('Points: ',Points3Dnew)

    Points3Dnew.map((v,i)=>{
        modelPoints.data64F[i] = v
    })

    console.log(modelPoints)

    const center = [width/2,height/2]

    const rvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
    const tvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);

    const imagePoints = cv.Mat.zeros(15, 2, cv.CV_64FC1);

    const pointZ = cv.matFromArray(1, 3, cv.CV_64FC1, [0.0, 0.0, 500.0]);

    const noseEndPoint2DZ = new cv.Mat();
    const nose_end_point2DY = new cv.Mat();
    const nose_end_point2DX = new cv.Mat();
    const jaco = new cv.Mat();
    

    const Points2D = []

    lm2DIndex.forEach(function(index){
        Points2D.push(landMarks2d[index].x);
        Points2D.push(landMarks2d[index].y)
    });

    console.log('2D Points: ',Points2D)

    const cameraMatrix = cv.matFromArray(3, 3, cv.CV_64FC1, [
        width,
        0,
        center[0] ,
        0,
        width,
        center[1] ,
        0,
        0,
        1
    ]);

    
    Points2D.map((v, i) => {
        imagePoints.data64F[i] = v;
    });

    console.log('image points:',imagePoints)

    const distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1);

    const success = await cv.solvePnP(
        modelPoints,
        imagePoints,
        cameraMatrix,
        distCoeffs,
        rvec,
        tvec,
        true
      );

    console.log("Rotation Vector:", rvec.data64F);

    console.log(
      "Rotation Vector (in degree):",
      rvec.data64F.map(d =>  (d / Math.PI) * 180)
    );

    console.log("Translation Vector:", tvec.data64F);
    
    cv.projectPoints(
        pointZ,
        rvec,
        tvec,
        cameraMatrix,
        distCoeffs,
        noseEndPoint2DZ,
        jaco
    );
    
      let im = cv.imread(document.querySelector("canvas"));
      // color the detected eyes and nose to purple
      for (var i = 0; i < 15; i++) {
        cv.circle(
          im,
          {
            x: imagePoints.doublePtr(i, 0)[0],
            y: imagePoints.doublePtr(i, 1)[0]
          },
          3,
          [255, 0, 255, 255],
          -1
        );
      }
      // draw axis
      const pNose = { x: imagePoints.data64F[0], y: imagePoints.data64F[1] };

      const pZ = {
        x: noseEndPoint2DZ.data64F[0],
        y: noseEndPoint2DZ.data64F[1]
      };

      if(pZ.x > pNose.x){
          pZ.x = pZ.x - (pZ.x - pNose.x) * 2
      }
      else{
          pZ.x = pZ.x + (pNose.x - pZ.x) * 2
      }
      cv.line(im, pNose, pZ, [255, 0, 0, 255], 2);

      //Display image
      cv.imshow(document.querySelector("canvas"), im);

}

