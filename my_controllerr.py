from controller import Robot, Motor, Gyro, GPS, Camera, Compass, Keyboard, LED, InertialUnit, DistanceSensor
import math
import cv2
import numpy as np
import time
from pyzbar.pyzbar import decode



SIGN = lambda x: int(x>0) - int(x<0)
CLAMP = lambda value, low, high : min(high, max(value, low))




class Drone:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        # front_left_led = robot.getDevice("front left led");
        # front_right_led = robot.getDevice("front right led");
        self.imu = self.robot.getDevice("inertial unit")
        self.imu.enable(self.timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(self.timestep)
        
        self.camera_roll_motor = self.robot.getDevice('camera roll')
        self.camera_pitch_motor = self.robot.getDevice('camera pitch')

        self.front_left_motor = self.robot.getDevice("front left propeller")
        self.front_right_motor = self.robot.getDevice("front right propeller")
        self.rear_left_motor = self.robot.getDevice("rear left propeller")
        self.rear_right_motor = self.robot.getDevice("rear right propeller")
        self.motors = [self.front_left_motor, self.front_right_motor, self.rear_left_motor, self.rear_right_motor]

        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1.0)

        self.k_vertical_thrust = 68.5
        self.k_vertical_offset = 0.6 
        self.k_vertical_p = 3.0
        self.k_roll_p = 50.0
        self.k_pitch_p = 30.0

        self.target_altitude = 1.0

    def move(self,command,intensity):
        roll = self.imu.getRollPitchYaw()[0] #+ math.pi / 2.0
        pitch = self.imu.getRollPitchYaw()[1]
        altitude = self.gps.getValues()[2]
        roll_acceleration = self.gyro.getValues()[0]
        pitch_acceleration = self.gyro.getValues()[1]
        

        # led_state = int(time) % 2
        # front_left_led.set(led_state)
        # front_right_led.set(int(not led_state))
        
        # self.camera_roll_motor.setPosition(-0.115 * roll_acceleration)
        # self.camera_pitch_motor.setPosition(-0.1 * pitch_acceleration)
        self.camera_roll_motor.setPosition(0)
        self.camera_pitch_motor.setPosition(math.pi/2)
        
        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0

        if(command=='forward'):
            pitch_disturbance = -intensity  #2.0
        elif(command=='backward'):
            pitch_disturbance = intensity #-2.0
        elif(command=='right'):
            yaw_disturbance = -intensity  #1.3
        elif(command=='left'):
            yaw_disturbance = intensity  #-1.3
        elif(command=='sRight'):
            roll_disturbance = -intensity  #-1.0
        elif(command=='sLeft'):
            roll_disturbance = intensity  #1.0
        elif(command=='up'):
            self.target_altitude += intensity  #0.05
        elif(command=='down'):
            self.target_altitude -= intensity  #0.05

        roll_input = self.k_roll_p * CLAMP(roll, -1.0, 1.0) + roll_acceleration + roll_disturbance
        pitch_input = self.k_pitch_p * CLAMP(pitch, -1.0, 1.0) + pitch_acceleration + pitch_disturbance
        yaw_input = yaw_disturbance
        clamped_difference_altitude = CLAMP(self.target_altitude - altitude + self.k_vertical_offset, -1.0, 1.0)
        vertical_input = self.k_vertical_p * pow(clamped_difference_altitude, 3.0)

        
        front_left_motor_input = self.k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
        front_right_motor_input = self.k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
        rear_left_motor_input = self.k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
        rear_right_motor_input = self.k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
        self.front_left_motor.setVelocity(front_left_motor_input)
        self.front_right_motor.setVelocity(-front_right_motor_input)
        self.rear_left_motor.setVelocity(-rear_left_motor_input)
        self.rear_right_motor.setVelocity(rear_right_motor_input)
        
    def get_image(self):
        image=self.camera.getImageArray()
        image=np.array(image,dtype=np.uint8)
        image = np.flip(image,axis=2)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = np.fliplr(image)
        return image
    
       






drone=Drone()


i=0
j=0
o=0
p=0
aa=[0,0]
aaa=[0,0]
count=0
cout=0
p=0
u=0
s=0
r=0
q=0
l=0

yy=0


RGB1="red"
RGB2="blue"
#rgb1=input()
#print("Enter rbg2")
#rgb2=input()

def c(rgb1):
        if(rgb1=="red"):
            c.lower=np.array([0,100,100])
            c.upper=np.array([10,255,255])
        if(rgb1=="yellow"):
            c.lower=np.array([20, 100, 100])
            c.upper=np.array([30, 255, 255])
        if(rgb1=="pink"):
            c.lower=np.array([140,100,100])
            c.upper=np.array([160,255,255])
        if(rgb1=="cyan"):
            c.lower=np.array([85,100,100])
            c.upper=np.array([95,255,255])
        if(rgb1=="blue"):
            c.lower=np.array([110,100,100])
            c.upper=np.array([120,255,255])
            
def ch(rgb2):
        if(rgb2=="red"):
            ch.lower=np.array([0,100,100])
            ch.upper=np.array([10,255,255])
        if(rgb2=="yellow"):
            ch.lower=np.array([20, 100, 100])
            ch.upper=np.array([30, 255, 255])
        if(rgb2=="pink"):
            ch.lower=np.array([140,100,100])
            ch.upper=np.array([160,255,255])
        if(rgb2=="cyan"):
            ch.lower=np.array([85,100,100])
            cc.upper=np.array([95,255,255])
        if(rgb2=="blue"):
            ch.lower=np.array([110,100,100])
            ch.upper=np.array([120,255,255])
        
c(RGB1)
ch(RGB2)
   


while drone.robot.step(drone.timestep) != -1:
    i=i+1
     
    if(i<100):
        drone.move('up',0.13)
    
    if(drone.gps.getValues()[0]<42  and drone.gps.getValues()[1]>(-15)) :   
        drone.move('forward',3)
       # print("3")  
        #print(drone.imu.getRollPitchYaw()[2])
    
    elif(drone.gps.getValues()[0]>=42):
       if(drone.imu.getRollPitchYaw()[2]>(-0.15)):
          drone.move('right',3)
         # print("1")  
          
       #if(drone.imu.getRollPitchYaw()[2]==-0.02737684834932314):
          # drone.move('forward',3)
       elif(drone.gps.getValues()[1]>(-15) and drone.imu.getRollPitchYaw()[2]<(-0.15)):
          drone.move('forward',3)
          #print("4")  
       #if(drone.gps.getValues()[1]>=(-15) and drone.imu.getRollPitchYaw()[2]<=(-0.15) and drone.imu.getRollPitchYaw()[2]>(-0.4)):
         # drone.move('right',3)
       elif(drone.imu.getRollPitchYaw()[2]>(-1.4)):
          drone.move('right',3)
          #print("2")  
          
       elif(drone.gps.getValues()[1]>(-26) and drone.imu.getRollPitchYaw()[2]>(-3)):
          drone.move('forward',3)
          #print("5")  
          
       elif(drone.imu.getRollPitchYaw()[2]>(-2.5)):
          drone.move('right',1)
          #print("7")  
       else:
          drone.move('forward',3)
         # print("6")  drone.move('forward',3)
     
    elif(drone.gps.getValues()[0]<33 and drone.gps.getValues()[1]<(-20) and i<9300):
          drone.move('sLeft',0.5) 
    elif(drone.gps.getValues()[0]<1 and drone.gps.getValues()[1]<(-20) and u==0):
          drone.move('forward',0)
    elif(u==1):
        if(drone.gps.getValues()[1]>y or yy==1):
            if(drone.gps.getValues()[0]>x and s==0 and q==0):
          
              drone.move('forward',1)
              #print("t")
            elif(drone.gps.getValues()[0]<=x):
          
              drone.move('backward',1)
              #print("tt")
              if(drone.gps.getValues()[0]>0.9*x):
                  s=1
            elif(drone.gps.getValues()[1]<=y and r==1 and q==0):
                drone.move('sRight',1)
                #print("tyt")
                q=1
            elif(drone.gps.getValues()[1]>y and r==1):
                drone.move('sLeft',1)
                #print("utt")
        
            elif(q==1):
            #if(l<100):
                drone.move('down',0.1) 
               # l=l+1 
            #else:
                #drone.move('down',1)   
            else:
              drone.move('up',0) 
             # print("y")
              r=1
            yy=1
        else:
            if(drone.gps.getValues()[0]>x and s==0 and q==0):
          
                  drone.move('forward',1)
                  #print("zt")
            elif(drone.gps.getValues()[0]<=x):
          
                  drone.move('backward',1)
                  #print("ztt")
                  if(drone.gps.getValues()[0]>0.9*x):
                      s=1
            elif(drone.gps.getValues()[1]>=y and r==1 and q==0):
                    drone.move('sLeft',1)
                    #print("ztyt")
                    q=1
            elif(drone.gps.getValues()[1]<y and r==1):
                    drone.move('sRight',0.5)
                    #print("zutt")
        
            elif(q==1):
            #if(l<100):
                drone.move('down',0.1) 
               # l=l+1 
            #else:
                #drone.move('down',1)   
            else:
              drone.move('up',0) 
              #print("zy")
              r=1
            yy==0
          
        #elif(drone.gps.getValues()[0]>0.9*x and drone.gps.getValues()[1]<y):
              # drone.move('sRight',1)
              # print("ty")
       # elif(drone.gps.getValues()[0]>0.9*x and drone.gps.getValues()[1]>y):
              # drone.move('sLeft',1)
              # print("yy")
        
    else:
         drone.move('forward',3) 
           
    #print(drone.gps.getValues()[0])   
    #print(drone.imu.getRollPitchYaw()[2]) 
    #print(i)   
   
    
    
    #if(drone.gps.getValues()[0]<45):    
       # drone.move('right',0.2)
        #print(drone.imu.getRollPitchYaw())
        
   # if(drone.gps.getValues()[0]>=45): 
       # while(drone.imu.getRollPitchYaw()[2]<0.785398):
          #  drone.move('right',0.2) 
       # drone.move('forward',3)
        
    
        
       
        
   # if(drone.gps.getValues()[0]>=2):    
        #drone.move('right',1)
        
        
        
            
            
                
       
    
        
    # print(drone.imu.getRollPitchYaw())
    # print(drone.gps.getValues())
    if (i%200==0 and i>1000):
        image=drone.get_image()
        lower=c.lower
        upper=c.upper
        inn=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(inn,lower,upper)
        ret, thresh1 = cv2.threshold(mask  , 120, 255, cv2.THRESH_BINARY)
        
        contours, hierarchy = cv2.findContours(thresh1, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
        lowerr=ch.lower
        upperr=ch.upper
        innr=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        maskk=cv2.inRange(innr,lowerr,upperr)
        ret, thresh1r = cv2.threshold(maskk  , 120, 255, cv2.THRESH_BINARY)
        
        cntours, hierarchy = cv2.findContours(thresh1r, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        
        
        if(len(contours)>0):
            a=1
        else:
            a=0
        #if(a==1):
            #print("Bomb Detected")
        if(aa[0]==0 and aa[1]==1):
            count=count+1   
        
        aa.insert(0,a)
        aa.pop()
        #print(aa)
        #print(count)
        #print("Number of Contours found = "+str(len(contours)))
        if(len(cntours)>0):
            w=1
        else:
            w=0
        #if(a==1):
            #print("Bomb Detected")
        if(aaa[0]==0 and aaa[1]==1):
            cout=cout+1   
        
        aaa.insert(0,w)
        aaa.pop()
        #print(aaa)
        #print(count+cout)
        
        X=abs(2*count-cout )
        #print(X)
        
        

        
        
        #cv2.imshow("img",mask)
        #cv2.waitKey(0)
        #cv2.imshow('Boundary', maskk)
        #cv2.waitKey(0)
       # cv2.imshow('Bo', t)
       # cv2.waitKey(0)
        cv2.destroyAllWindows()
        if(drone.gps.getValues()[1]<-25 ):
            imagee=drone.get_image()
            ret, thresh2 = cv2.threshold(imagee, 120, 255, cv2.THRESH_BINARY)
            #cv2.imshow('Binary Threshold Inverted', thresh2) 
            #cv2.waitKey(0)
            code=decode(imagee)
            #print(len(code))
            if(len(code)==4 and p==0):
            
                for barcode in code:
                
                #print(barcode.data)
                    pp=barcode.data.decode('utf-8')
                    
                    data=[e.strip() for e in pp.split(',')]
                    #print(pp)
                    #print(data[2])
                    
                    if(int(data[0])==X):
                        x=float(data[1])
                        y=float(data[2])
                        print(x)
                        print(y)
                        u=1
                p=1
              
             
            
    # image=drone.get_image()
    
   
    # cv2.imshow("img",image)
    # if cv2.waitKey() & 0xFF == ord('q'):
        # break

cv2.destroyAllWindows()

    


   


