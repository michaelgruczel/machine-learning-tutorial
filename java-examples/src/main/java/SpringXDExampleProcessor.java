public class SpringXDExampleProcessor {

    public String transform(String payload) {
        if(payload.contains("good") && payload.contains("bad")) {
            return payload + "->neutral";
        } else if(payload.contains("good")) {
            return payload + "->good";
        } else if(payload.contains("bad")) {
            return payload + "->bad";
        } else {
            return payload + "->neutral";
        }
    }

}
