const lmType = 1

const landmarks3DList = []

const component1 = tf.tensor2d([
    [ 0.000,  0.000,   0.000],    
    [ 0.000, -8.250,  -1.625],   
    [-5.625,  4.250,  -3.375],   
    [ 5.625,  4.250,  -3.375],  
    [-3.750, -3.750,  -3.125], 
    [ 3.750, -3.750,  -3.125] 
])

const component2 = tf.tensor2d([
    [ 0.000000,  0.000000,  6.763430],
    [ 6.825897,  6.760612,  4.402142], 
    [ 1.330353,  7.122144,  6.903745],
    [-1.330353,  7.122144,  6.903745],
    [-6.825897,  6.760612,  4.402142],
    [ 5.311432,  5.485328,  3.987654],
    [ 1.789930,  5.393625,  4.413414], 
    [-1.789930,  5.393625,  4.413414],
    [-5.311432,  5.485328,  3.987654],
    [ 2.005628,  1.409845,  6.165652],
    [-2.005628,  1.409845,  6.165652],
    [ 2.774015, -2.080775,  5.048531],
    [-2.774015, -2.080775,  5.048531],
    [ 0.000000, -3.116408,  6.097667],
    [ 0.000000, -7.415691,  4.070434] 
])

const component3 = tf.tensor2d([
    [ 0.000000,  0.000000,  6.763430],
    [ 5.311432,  5.485328,  3.987654],
    [ 1.789930,  5.393625,  4.413414],
    [-1.789930,  5.393625,  4.413414],
    [-5.311432,  5.485328,  3.987654]
])


landmarks3DList.push(component1);
landmarks3DList.push(component2);
landmarks3DList.push(component3)

const landmarks2DList = [
    [30,8,36,45,48,54],
    [33,17,21,22,26,36,39,42,45,31,48,54,57,8],
    [33,36,39,3,42,45]
]

const lm2DIndex = landmarks2DList[lmType];
const landmarks3D = landmarks3DList[lmType];

const toTensor = function(landMarks){
    coords = []
    landMarks.forEach(point => {
        coords.push([point.x,point.y])
    });
    coords = tf.tensor2d(coords);
    return coords
}

const getHeadpose = function(image,landMarks2d,verbose=false){


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
    

    const ns = landMarks2d[30];
    const le = landMarks2d[36];
    const c = landMarks2d[8]
    const re = landMarks2d[45];
    const lm = landMarks2d[48];
    const rm = landMarks2d[54];

    

    // distCoeffs = tf.zeros([4,1]);

    const modelPoints = cv.Mat.zeros(6, 3, cv.CV_64FC1);

    console.log(cv.solvePnP);

    const Points3D = [0.0,0.0,0.0,0.0,-330.0,-65.0,-225,170.0,-135.0,225.0,170.0,-135.0,225.0,170.0,-135.0,-150.0,-150.0,-125.0,150.0,-150.0,-125.0]


    Points3D.map((v,i)=>{
        modelPoints.data64F[i] = v
    })

    console.log(modelPoints)

    // [
    //     0.0,
    //     0.0,
    //     0.0, // Nose tip

    //     0.0,
    //     -330,0,
    //     -65.0, // HACK! solvePnP doesn't work with 3 points, so copied the
    //     //   first point to make the input 4 points
    //     // 0.0, -330.0, -65.0,  // Chin
    //     -225.0,
    //     170.0,
    //     -135.0, // Left eye left corner

    //     225.0,
    //     170.0,
    //     -135.0, // Right eye right corne

    //      -150.0, -150.0, -125.0,  // Left Mouth corner
    //      150.0, -150.0, -125.0  // Right mouth corner
    //   ].map((v,i)=>{
    //     modelPoints.data64F[i] = v;
    //   })

      console.log(modelPoints)

      const center = [width/2,height/2]

      const rvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
      const tvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);

      const imagePoints = cv.Mat.zeros(6, 2, cv.CV_64FC1);

      const pointZ = cv.matFromArray(1, 3, cv.CV_64FC1, [0.0, 0.0, 1000.0]);
      const pointY = cv.matFromArray(1, 3, cv.CV_64FC1, [0.0, 1000.0, 0.0]);
      const pointX = cv.matFromArray(1, 3, cv.CV_64FC1, [1000.0, 0.0, 0.0]);
      const noseEndPoint2DZ = new cv.Mat();
      const nose_end_point2DY = new cv.Mat();
      const nose_end_point2DX = new cv.Mat();
      const jaco = new cv.Mat();

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

    [
        ns.x,
        ns.y, // Nose tip
        c.x,
        c.y, // Nose tip (see HACK! above)
        // 399, 561, // Chin
        le.x,
        le.y, // Left eye left corner
        re.x,
        re.y, // Right eye right corner
        lm.x, lm.y, // Left Mouth corner
        rm.x, rm.y // Right mouth corner
      ].map((v, i) => {
        imagePoints.data64F[i] = v;
      });

      console.log(imagePoints)

      tvec.data64F[0] = -100;
      tvec.data64F[1] = 100;
      tvec.data64F[2] = 1000;
      const distToLeftEyeX = Math.abs(le.x - ns.x);
      const distToRightEyeX = Math.abs(re.x - ns.x);
      if (distToLeftEyeX < distToRightEyeX) {
        // looking at left
        rvec.data64F[0] = -1.0;
        rvec.data64F[1] = -0.75;
        rvec.data64F[2] = -3.0;
      } else {
        // looking at right
        rvec.data64F[0] = 1.0;
        rvec.data64F[1] = -0.75;
        rvec.data64F[2] = -3.0;
      }


    const distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1);

    const success = cv.solvePnP(
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
      rvec.data64F.map(d => (d / Math.PI) * 180)
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
    //   cv.projectPoints(
    //     pointY,
    //     rvec,
    //     tvec,
    //     cameraMatrix,
    //     distCoeffs,
    //     nose_end_point2DY,
    //     jaco
    //   );
    //   cv.projectPoints(
    //     pointX,
    //     rvec,
    //     tvec,
    //     cameraMatrix,
    //     distCoeffs,
    //     nose_end_point2DX,
    //     jaco
    //   );
      let im = cv.imread(document.querySelector("canvas"));
      // color the detected eyes and nose to purple
      for (var i = 0; i < 6; i++) {
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
    //   const p3 = {
    //     x: nose_end_point2DY.data64F[0],
    //     y: nose_end_point2DY.data64F[1]
    //   };
    //   const p4 = {
    //     x: nose_end_point2DX.data64F[0],
    //     y: nose_end_point2DX.data64F[1]
    //   };
      cv.line(im, pNose, pZ, [255, 0, 0, 255], 2);
    //   cv.line(im, pNose, p3, [0, 255, 0, 255], 2);
    //   cv.line(im, pNose, p4, [0, 0, 255, 255], 2);

      // Display image
      cv.imshow(document.querySelector("canvas"), im);

}

