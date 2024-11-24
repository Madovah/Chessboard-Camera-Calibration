import numpy as np
import cv2 as cv


def select_input(input, board_size, selct_all=False, pause=10):
    video_input = cv.VideoCapture(input)
    assert video_input.isOpened(), 'Cannot read the given input, ' + input

    selections = []
    while True:

        validating, img = video_input.read()
        if not validating:
            break

        if selct_all:
            selections.append(img)
        else:

            display = img.copy()
            cv.putText(display, f'Selected_frames: {len(selections)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow('Camera Calibration', display)

            # Process the key event
            key = cv.waitKey(pause)
            if key == 27:
                break
            elif key == ord(' '):
                res, pts = cv.findChessboardCorners(img, board_size)
                cv.drawChessboardCorners(display, board_size, pts, res)
                cv.imshow('Camera Calibration', display)
                print(f'Selected_frames: {len(selections)}')
                key = cv.waitKey()
                if key == 27:
                    break
                elif key == ord('s'):
                    selections.append(img)

    cv.destroyAllWindows()
    return selections


def calib_camera(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):

    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv. COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)

    assert len(img_points) > 0, 'There is no set of complete chessboard points!'

    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)  # it has to be np.float32
    print(np.array(obj_pts).shape)

    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags) # Calibration


input_file = 'vid.mp4'
pattern = (7, 7)
celsize = 0.01

selected = select_input(input_file, pattern)
print(pattern, celsize)
assert len(selected) > 0, 'No selections were given!!'
rms, K, distorsion_coeffs, rvects, tvects = calib_camera(selected, pattern, celsize)


print('---- Calibration Results ----')
print(f'-- The number of selections = {len(selected)}')
print(f'-- RMS error = {rms}')
print(f'-- Cam matrix (K) = \n{K}')
print(f'-- Distortion coefficient (k1, k2, p1, p2, k3, ...) = {distorsion_coeffs.flatten()}')

criter = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK


vid = cv.VideoCapture(input_file)
assert vid.isOpened(), 'Error reading input... ' + input_file


lower_b = celsize * np.array([[4, 2, 0], [5, 2, 0], [5, 4, 0], [4, 4, 0]])
upper_b = celsize * np.array([[4, 2, -1], [5, 2, -1], [5, 4, -1], [4, 4, -1]])
below_cir = celsize * np.array([4, 2, 0])

obj_points = celsize * np.array([[c, r, 0] for r in range(pattern[1]) for c in range(pattern[0])])


while True:
    valid, img = vid.read()
    if not valid:
        break

    complete, img_points = cv.findChessboardCorners(img, pattern, criter)
    if complete:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, distorsion_coeffs)

        line_lower, _ = cv.projectPoints(lower_b, rvec, tvec, K, distorsion_coeffs)
        line_upper, _ = cv.projectPoints(upper_b, rvec, tvec, K, distorsion_coeffs)
        circle_lower, _ = cv.projectPoints(below_cir, rvec, tvec, K, distorsion_coeffs)
        cv.circle(img, circle_lower.squeeze().astype(np.int32), radius=30, color=(0, 255, 0), thickness=-1)

        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'Axes: X,Y,Z: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    cv.imshow('Pose Estimation on the board', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:
        break

vid.release()
cv.destroyAllWindows()
