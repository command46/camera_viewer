public class Run {
    public static void main(String[] args) {
       Thread videoServerThread = new Thread(() -> {
            VideoServer.main(args);
        });
        videoServerThread.start();
        Thread photoServerThread = new Thread(() -> {
            PhotoServer.main(args);
        });
        photoServerThread.start();
    }
}
