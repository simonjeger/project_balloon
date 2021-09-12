//Initializing LED Pin
int led_pin = 13;
int pwm_pin = 2;
void setup() {
  //Declaring LED pin as output
  pinMode(led_pin, OUTPUT);
  pinMode(pwm_pin, OUTPUT);
}
void loop() {
  //Fading the LED
  for(int i=0; i<255; i++){
    analogWrite(led_pin, i);
    analogWrite(pwm_pin, i);
    delay(10);
  }
  for(int i=255; i>0; i--){
    analogWrite(led_pin, i);
    analogWrite(pwm_pin, i);
    delay(10);
  }
}
