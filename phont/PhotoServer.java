import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.nio.file.*;

public class PhotoServer {
    private static final int PORT = 12345;
    private static final String BASE_DIR = "captured_photos";
    private static final int MAX_FILENAME_LENGTH = 255;

    public static void main(String[] args) {
        System.out.println("正在创建目录结构...");
        createBaseDirectory();

        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("服务器启动，监听端口: " + PORT);

            while (true) {
                Socket clientSocket = null;
                try {
                    clientSocket = serverSocket.accept();
                    System.out.println("收到新的连接: " + clientSocket.getInetAddress());
                    handleClient(clientSocket);
                } catch (IOException e) {
                    System.err.println("处理客户端连接时发生错误: " + e.getMessage());
                    e.printStackTrace();
                } finally {
                    if (clientSocket != null && !clientSocket.isClosed()) {
                        try {
                            clientSocket.close();
                        } catch (IOException e) {
                            System.err.println("关闭socket时发生错误: " + e.getMessage());
                        }
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("服务器启动失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void createBaseDirectory() {
        try {
            Path basePath = Paths.get(BASE_DIR).toAbsolutePath();
            System.out.println("基础目录路径: " + basePath);

            if (!Files.exists(basePath)) {
                Files.createDirectories(basePath);
                System.out.println("创建基础目录成功");
            }

            // No need to create today's directory here, it will be created in handleClient
        } catch (IOException e) {
            System.err.println("创建目录结构失败: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void handleClient(Socket clientSocket) throws IOException {
        DataInputStream dis = null;
        OutputStream fos = null;

        try {
            dis = new DataInputStream(new BufferedInputStream(clientSocket.getInputStream()));

            // 读取文件名并验证
            String fileName = dis.readUTF();
            System.out.println("接收到的文件名: [" + fileName + "]");

            if (fileName == null || fileName.trim().isEmpty()) {
                System.err.println("文件名为空，跳过处理");
                return;
            }

            if (fileName.length() > MAX_FILENAME_LENGTH) {
                System.err.println("文件名过长，跳过处理");
                return;
            }

            // 清理文件名，移除任何路径分隔符
            fileName = fileName.replace('/', '_').replace('\\', '_');

            // 准备保存路径
            String today = new SimpleDateFormat("yyyy-MM-dd").format(new Date());
            Path basePath = Paths.get(BASE_DIR).toAbsolutePath();
            Path todayPath = basePath.resolve(today);

            // 根据文件名中的 front 或 back 关键字创建子文件夹
            String subfolderName = fileName.contains("front") ? "front" : "back";
            Path subfolderPath = todayPath.resolve(subfolderName);
            
            Path filePath = subfolderPath.resolve(fileName);

            System.out.println("准备保存文件到: " + filePath);

            // 确保目标目录存在
            Files.createDirectories(subfolderPath);

            // 读取文件内容
            fos = Files.newOutputStream(filePath, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
            byte[] buffer = new byte[8192];
            int bytesRead;
            long totalBytes = 0;

            while ((bytesRead = dis.read(buffer)) != -1) {
                fos.write(buffer, 0, bytesRead);
                totalBytes += bytesRead;
                // 简单的进度显示
                if (totalBytes % (1024 * 1024) == 0) { // 每MiB显示一次
                    System.out.println("已接收: " + (totalBytes / (1024 * 1024)) + " MiB");
                }
            }

            System.out.println("文件保存成功: " + filePath);
            System.out.println("总大小: " + totalBytes + " bytes");

        } catch (IOException e) {
            System.err.println("文件传输失败: " + e.getMessage());
            throw e;
        } finally {
            // 关闭资源
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    System.err.println("关闭文件输出流失败: " + e.getMessage());
                }
            }
            if (dis != null) {
                try {
                    dis.close();
                } catch (IOException e) {
                    System.err.println("关闭数据输入流失败: " + e.getMessage());
                }
            }
        }
    }
}